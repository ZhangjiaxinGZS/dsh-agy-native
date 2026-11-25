import { ToolCallId } from '@deepseek-ai/dsh-llm';
import type {
  GenerateOptions,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm';
import { AcpWorkerClient, type AcpAgentCapabilitiesSummary } from './acp-client.js';
import { resolveAgyBinary, supportsProcessPerSession } from './binary.js';
import type { AgyModelCatalog } from './catalog.js';
import {
  computeToolsFingerprint,
  type LoopbackMcpBridge,
  type McpSessionEndpoint,
} from './mcp-bridge.js';
import {
  buildFullReplayPromptParts,
  buildIncrementalPromptParts,
  computeContextDigests,
  computeHistoryPrefixDigest,
  extractReplayMetadata,
  extractSystemPrompt,
  formatHarnessGuidelines,
  planSessionSync,
  type ImageBytesLoader,
} from './session-sync.js';
import type {
  AcpPromptResponse,
  AcpPromptUsage,
  AgyReplayMetadata,
  PendingMcpToolCall,
  ResolvedAgyAcpConfig,
} from './types.js';

type TurnQueueEvent =
  | { kind: 'thought'; text: string }
  | { kind: 'text'; text: string }
  | { kind: 'tool-batch'; calls: PendingMcpToolCall[] }
  | { kind: 'prompt-done'; response: AcpPromptResponse }
  | { kind: 'error'; error: Error }
  | { kind: 'aborted'; reason: Error };

class AsyncEventQueue<T> {
  readonly #items: T[] = [];
  readonly #waiters: Array<(result: IteratorResult<T>) => void> = [];
  #closed = false;

  push(item: T): void {
    if (this.#closed) return;
    const waiter = this.#waiters.shift();
    if (waiter) {
      waiter({ value: item, done: false });
    } else {
      this.#items.push(item);
    }
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    while (this.#waiters.length > 0) {
      const waiter = this.#waiters.shift()!;
      waiter({ value: undefined as unknown as T, done: true });
    }
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.#items.length > 0) {
      return { value: this.#items.shift()!, done: false };
    }
    if (this.#closed) {
      return { value: undefined as unknown as T, done: true };
    }
    return new Promise<IteratorResult<T>>((resolve) => {
      this.#waiters.push(resolve);
    });
  }
}

export function mapAcpUsageToDsh(usage?: AcpPromptUsage): TokenUsage | undefined {
  if (!usage) return undefined;
  const inputTokens = Math.max(0, usage.inputTokens ?? 0);
  const outputTokens = Math.max(0, usage.outputTokens ?? 0);
  const cacheReadTokens = usage.cachedReadTokens;
  const cacheWriteTokens = usage.cachedWriteTokens;
  const reasoningTokens = usage.thoughtTokens;
  const totalTokens =
    usage.totalTokens ??
    inputTokens + outputTokens + (cacheReadTokens ?? 0) + (cacheWriteTokens ?? 0);

  return {
    inputTokens,
    outputTokens,
    totalTokens,
    ...(typeof cacheReadTokens === 'number' ? { cacheReadTokens } : {}),
    ...(typeof cacheWriteTokens === 'number' ? { cacheWriteTokens } : {}),
    ...(typeof reasoningTokens === 'number' ? { reasoningTokens } : {}),
  };
}

/**
 * One stateful ACP worker slot holding a persistent `agy acp --stdio` child process
 * and its active conversation session.
 */
export class AgyWorkerSlot {
  readonly id: string;
  readonly #getConfig: () => ResolvedAgyAcpConfig;
  readonly #mcpBridge: LoopbackMcpBridge;
  readonly #catalog: AgyModelCatalog;
  readonly #loadImage?: ImageBytesLoader;

  #client?: AcpWorkerClient;
  #endpoint?: McpSessionEndpoint;
  #activeAcpSessionId?: string;
  #activeRawModelId?: string;
  #lastToolsFingerprint?: string;
  #lastContextDigests?: Record<string, string>;
  #lastHistoryPrefixDigest?: string;
  #inFlightPrompt?: Promise<AcpPromptResponse>;
  #inFlightError?: Error;

  #busy = false;
  #lastUsedAt = Date.now();
  #idleTimer?: NodeJS.Timeout;
  #boundKey?: string;

  constructor(opts: {
    id: string;
    getConfig: () => ResolvedAgyAcpConfig;
    mcpBridge: LoopbackMcpBridge;
    catalog: AgyModelCatalog;
    loadImage?: ImageBytesLoader;
  }) {
    this.id = opts.id;
    this.#getConfig = opts.getConfig;
    this.#mcpBridge = opts.mcpBridge;
    this.#catalog = opts.catalog;
    this.#loadImage = opts.loadImage;
  }

  get busy(): boolean {
    return this.#busy;
  }

  get hasPendingToolTurn(): boolean {
    return Boolean(this.#inFlightPrompt && (this.#endpoint?.pendingCount ?? 0) > 0);
  }

  get lastUsedAt(): number {
    return this.#lastUsedAt;
  }

  get boundKey(): string | undefined {
    return this.#boundKey;
  }

  setBoundKey(key: string | undefined): void {
    this.#boundKey = key;
  }

  get activeAcpSessionId(): string | undefined {
    return this.#activeAcpSessionId;
  }

  /** Ensure `agy` is started and fetch catalog models via a lightweight probe session if needed. */
  async probeCatalog(): Promise<AcpAgentCapabilitiesSummary> {
    const cfg = this.#getConfig();
    const client = await this.#ensureClient(cfg);
    const caps = await client.start();
    if (!this.#catalog.snapshot) {
      const sess = await client.createSession({ cwd: cfg.workingDirectory });
      if (sess.availableModels.length > 0) {
        this.#catalog.update(sess.availableModels, sess.tier);
      }
      await client.closeSession(sess.sessionId);
    }
    this.#scheduleIdleTimer(cfg.idleTtlMs);
    return caps;
  }

  async *streamTurn(options: GenerateOptions<string>): AsyncGenerator<StreamChunk> {
    if (options.signal?.aborted) {
      throw options.signal.reason ?? new Error('Request aborted before start');
    }

    this.#clearIdleTimer();
    this.#busy = true;
    this.#lastUsedAt = Date.now();

    const cfg = this.#getConfig();
    const targetRawModelId = this.#catalog.resolveRawModelId(
      options.model,
      options.reasoningEffort,
    );
    const toolsFp = computeToolsFingerprint(options.tools);
    const systemPrompt = extractSystemPrompt(options.messages, options.system);

    try {
      const client = await this.#ensureClient(cfg);
      const endpoint = await this.#ensureEndpoint(cfg);
      endpoint.setTools(options.tools);
      client.setActiveEndpoint(endpoint);

      const pendingIds = new Set(endpoint.getPendingCalls().map((c) => c.toolCallId));
      const syncPlan = planSessionSync({
        messages: options.messages,
        activeAcpSessionId: this.#activeAcpSessionId,
        lastToolsFingerprint: this.#lastToolsFingerprint,
        currentToolsFingerprint: toolsFp,
        lastContextDigests: this.#lastContextDigests,
        lastHistoryPrefixDigest: this.#lastHistoryPrefixDigest,
        pendingToolCallIds: pendingIds,
        allowResumeFromReplay: cfg.resumeSessions,
        forwardContextUpdates: cfg.forwardContextUpdates,
      });

      const queue = new AsyncEventQueue<TurnQueueEvent>();
      const detachBatch = endpoint.onNextBatch((calls) => {
        queue.push({ kind: 'tool-batch', calls });
      });

      client.setStreamCallbacks({
        onThoughtDelta: (text) => queue.push({ kind: 'thought', text }),
        onTextDelta: (text) => queue.push({ kind: 'text', text }),
        onUsageUpdate: (_used, size) => {
          this.#catalog.recordContextWindow(targetRawModelId, size);
        },
      });

      const onAbort = () => {
        const reason =
          options.signal?.reason instanceof Error
            ? options.signal.reason
            : new Error(String(options.signal?.reason ?? 'Request aborted'));
        endpoint.cancelAllPending(reason.message);
        if (this.#activeAcpSessionId) {
          client.cancelPrompt(this.#activeAcpSessionId);
        }
        queue.push({ kind: 'aborted', reason });
      };
      options.signal?.addEventListener('abort', onAbort, { once: true });

      try {
        if (syncPlan.kind === 'resolve-pending-tools' && this.#inFlightPrompt) {
          if (this.#inFlightError) {
            const staleErr = this.#inFlightError;
            this.#inFlightPrompt = undefined;
            this.#inFlightError = undefined;
            throw staleErr;
          }
          this.#attachPromptCompletion(this.#inFlightPrompt, queue);
          for (const res of syncPlan.resolutions) {
            endpoint.resolveToolCall(res.toolCallId, {
              text: res.text,
              isError: res.isError,
            });
          }
        } else {
          // Tear down any abandoned in-flight turn before starting a fresh prompt
          await this.#abortInFlightTurn(syncPlan.kind !== 'warm-incremental');

          let promptParts;
          if (syncPlan.kind === 'warm-incremental' && this.#activeAcpSessionId) {
            if (this.#activeRawModelId !== targetRawModelId) {
              await client.setModel(this.#activeAcpSessionId, targetRawModelId);
              this.#activeRawModelId = targetRawModelId;
            }
            promptParts = await buildIncrementalPromptParts({
              tailMessages: syncPlan.tailMessages,
              updatedContextBlocks: syncPlan.updatedContextBlocks,
              loadImage: this.#loadImage,
              signal: options.signal,
            });
          } else if (syncPlan.kind === 'resume-incremental') {
            let resumed = false;
            if (client.capabilities?.loadSession) {
              try {
                await client.loadSession({
                  sessionId: syncPlan.resumeAcpSessionId,
                  endpoint,
                  cwd: cfg.workingDirectory,
                });
                await client.setModel(syncPlan.resumeAcpSessionId, targetRawModelId);
                this.#activeAcpSessionId = syncPlan.resumeAcpSessionId;
                this.#activeRawModelId = targetRawModelId;
                resumed = true;
              } catch {
                resumed = false;
              }
            }
            if (resumed) {
              promptParts = await buildIncrementalPromptParts({
                tailMessages: syncPlan.tailMessages,
                updatedContextBlocks: syncPlan.updatedContextBlocks,
                loadImage: this.#loadImage,
                signal: options.signal,
              });
              // Loaded sessions may have been created by an older adapter.
              const routing = formatHarnessGuidelines(undefined, endpoint.bridgeToolNames);
              if (routing) promptParts.unshift({ type: 'text', text: routing });
            } else {
              await this.#openFreshSession(client, endpoint, targetRawModelId, cfg);
              promptParts = await buildFullReplayPromptParts({
                messages: options.messages,
                systemPrompt,
                bridgeToolNames: endpoint.bridgeToolNames,
                loadImage: this.#loadImage,
                signal: options.signal,
              });
            }
          } else {
            await this.#openFreshSession(client, endpoint, targetRawModelId, cfg);
            promptParts = await buildFullReplayPromptParts({
              messages: options.messages,
              systemPrompt,
              bridgeToolNames: endpoint.bridgeToolNames,
              loadImage: this.#loadImage,
              signal: options.signal,
            });
          }

          const promptPromise = client.prompt(
            this.#activeAcpSessionId!,
            promptParts,
            cfg.requestTimeoutMs > 0 ? cfg.requestTimeoutMs : undefined,
          );
          // Prevent unhandled rejection if `agy` errors while suspended between DSH turns
          this.#inFlightError = undefined;
          promptPromise.catch((err: unknown) => {
            this.#inFlightError = err instanceof Error ? err : new Error(String(err));
          });
          this.#inFlightPrompt = promptPromise;
          this.#attachPromptCompletion(promptPromise, queue);
        }

        // Stream events out as DSH StreamChunks
        let nextBlockIndex = 0;
        let openBlock:
          | { index: number; type: 'reasoning' | 'text'; text: string }
          | undefined;

        const closeOpenBlock = (): StreamChunk | undefined => {
          if (!openBlock) return undefined;
          const chunk: StreamChunk = {
            type: 'block-end',
            index: openBlock.index,
            block:
              openBlock.type === 'reasoning'
                ? { type: 'reasoning', text: openBlock.text }
                : { type: 'text', text: openBlock.text },
          };
          openBlock = undefined;
          return chunk;
        };

        const buildReplayEnvelope = () => {
          const historyPrefixDigest = computeHistoryPrefixDigest(
            options.messages,
            options.messages.length,
          );
          const meta: AgyReplayMetadata = {
            version: 1,
            acpSessionId: this.#activeAcpSessionId ?? '',
            rawModelId: targetRawModelId,
            syncedMessageCount: options.messages.length + 1,
            toolsFingerprint: toolsFp,
            contextDigests: computeContextDigests(options.messages),
            historyPrefixDigest,
          };
          this.#lastToolsFingerprint = toolsFp;
          this.#lastContextDigests = meta.contextDigests;
          this.#lastHistoryPrefixDigest = historyPrefixDigest;
          return { response: meta };
        };

        while (true) {
          const step = await queue.next();
          if (step.done) break;
          const ev = step.value;

          if (ev.kind === 'thought') {
            if (openBlock && openBlock.type !== 'reasoning') {
              yield closeOpenBlock()!;
            }
            if (!openBlock) {
              openBlock = { index: nextBlockIndex++, type: 'reasoning', text: '' };
              yield { type: 'block-start', index: openBlock.index, blockType: 'reasoning' };
            }
            openBlock.text += ev.text;
            yield { type: 'reasoning-delta', index: openBlock.index, text: ev.text };
            continue;
          }

          if (ev.kind === 'text') {
            if (openBlock && openBlock.type !== 'text') {
              yield closeOpenBlock()!;
            }
            if (!openBlock) {
              openBlock = { index: nextBlockIndex++, type: 'text', text: '' };
              yield { type: 'block-start', index: openBlock.index, blockType: 'text' };
            }
            openBlock.text += ev.text;
            yield { type: 'text-delta', index: openBlock.index, text: ev.text };
            continue;
          }

          if (ev.kind === 'tool-batch') {
            const closed = closeOpenBlock();
            if (closed) yield closed;

            for (const call of ev.calls) {
              const idx = nextBlockIndex++;
              const callId = ToolCallId(call.toolCallId);
              yield { type: 'block-start', index: idx, blockType: 'tool-call' };
              yield {
                type: 'tool-call-delta',
                index: idx,
                id: callId,
                name: call.toolName,
                argumentsDelta: call.argumentsJson,
              };
              yield {
                type: 'block-end',
                index: idx,
                block: {
                  type: 'tool-call',
                  id: callId,
                  name: call.toolName,
                  arguments: call.argumentsJson,
                },
              };
            }

            yield {
              type: 'finish',
              reason: { kind: 'tool-calls' },
              replayState: buildReplayEnvelope(),
            };
            queue.close();
            break;
          }

          if (ev.kind === 'prompt-done') {
            this.#inFlightPrompt = undefined;
            this.#inFlightError = undefined;
            const closed = closeOpenBlock();
            if (closed) yield closed;

            const usage = mapAcpUsageToDsh(ev.response.usage);
            if (usage) {
              yield { type: 'usage', usage };
            }

            const stopReason =
              ev.response.stopReason === 'max_tokens'
                ? ({ kind: 'max-tokens' } as const)
                : ({ kind: 'stop' } as const);

            yield {
              type: 'finish',
              reason: stopReason,
              replayState: buildReplayEnvelope(),
            };
            queue.close();
            break;
          }

          if (ev.kind === 'aborted') {
            this.#inFlightPrompt = undefined;
            this.#inFlightError = undefined;
            queue.close();
            throw ev.reason;
          }

          if (ev.kind === 'error') {
            this.#inFlightPrompt = undefined;
            this.#inFlightError = undefined;
            queue.close();
            throw ev.error;
          }
        }
      } finally {
        options.signal?.removeEventListener('abort', onAbort);
        detachBatch();
        client.setStreamCallbacks(undefined);
      }
    } finally {
      this.#busy = false;
      this.#lastUsedAt = Date.now();
      this.#scheduleIdleTimer(cfg.idleTtlMs);
    }
  }

  async dispose(): Promise<void> {
    this.#clearIdleTimer();
    this.#endpoint?.dispose();
    this.#endpoint = undefined;
    this.#inFlightPrompt = undefined;
    this.#inFlightError = undefined;
    this.#clearActiveSessionState();
    const client = this.#client;
    this.#client = undefined;
    if (client) {
      await client.dispose();
    }
  }

  async #ensureClient(cfg: ResolvedAgyAcpConfig): Promise<AcpWorkerClient> {
    if (this.#client?.alive) return this.#client;
    if (this.#client) {
      await this.#client.dispose();
    }
    this.#clearActiveSessionState();
    this.#inFlightPrompt = undefined;
    this.#inFlightError = undefined;
    const agyPath = resolveAgyBinary(cfg.agyPath);
    this.#client = new AcpWorkerClient({
      agyPath,
      extraArgs: await supportsProcessPerSession(agyPath) ? ['--process-per-session'] : [],
      cwd: cfg.workingDirectory,
      startupTimeoutMs: cfg.startupTimeoutMs,
    });
    await this.#client.start();
    return this.#client;
  }

  async #ensureEndpoint(cfg: ResolvedAgyAcpConfig): Promise<McpSessionEndpoint> {
    if (this.#endpoint) return this.#endpoint;
    this.#endpoint = await this.#mcpBridge.createSessionEndpoint({
      maxToolOutputBytes: cfg.maxToolOutputBytes,
      batchWindowMs: cfg.batchWindowMs,
    });
    return this.#endpoint;
  }

  async #openFreshSession(
    client: AcpWorkerClient,
    endpoint: McpSessionEndpoint,
    targetRawModelId: string,
    cfg: ResolvedAgyAcpConfig,
  ): Promise<void> {
    if (this.#activeAcpSessionId) {
      await client.closeSession(this.#activeAcpSessionId);
      this.#clearActiveSessionState();
    }

    const created = await client.createSession({
      endpoint,
      cwd: cfg.workingDirectory,
    });

    if (created.availableModels.length > 0) {
      this.#catalog.update(created.availableModels, created.tier);
    }

    const modelAvailable =
      created.availableModels.length === 0 ||
      created.availableModels.some((m) => m.modelId === targetRawModelId);
    if (!modelAvailable) {
      throw new Error(
        `dsh-agy-native: model '${targetRawModelId}' is not available on this Google subscription (${created.tier?.name ?? 'unknown tier'}). Available: ${created.availableModels.map((m) => m.modelId).join(', ')}`,
      );
    }

    if (created.currentModelId !== targetRawModelId) {
      await client.setModel(created.sessionId, targetRawModelId);
    }

    this.#activeAcpSessionId = created.sessionId;
    this.#activeRawModelId = targetRawModelId;
  }

  async #abortInFlightTurn(closeExistingSession: boolean): Promise<void> {
    this.#endpoint?.cancelAllPending('Turn superseded');
    if (this.#inFlightPrompt && this.#activeAcpSessionId && this.#client?.alive) {
      this.#client.cancelPrompt(this.#activeAcpSessionId);
      await Promise.race([
        this.#inFlightPrompt.catch(() => undefined),
        new Promise<void>((r) => setTimeout(r, 500)),
      ]);
    }
    this.#inFlightPrompt = undefined;
    this.#inFlightError = undefined;
    if (closeExistingSession) {
      if (this.#activeAcpSessionId && this.#client?.alive) {
        await this.#client.closeSession(this.#activeAcpSessionId);
      }
      this.#clearActiveSessionState();
    }
  }

  #clearActiveSessionState(): void {
    this.#activeAcpSessionId = undefined;
    this.#activeRawModelId = undefined;
    this.#lastToolsFingerprint = undefined;
    this.#lastContextDigests = undefined;
    this.#lastHistoryPrefixDigest = undefined;
  }

  #attachPromptCompletion(
    promptPromise: Promise<AcpPromptResponse>,
    queue: AsyncEventQueue<TurnQueueEvent>,
  ): void {
    promptPromise.then(
      (response) => {
        queue.push({ kind: 'prompt-done', response });
      },
      (err: unknown) => {
        const error = err instanceof Error ? err : new Error(String(err));
        queue.push({ kind: 'error', error });
      },
    );
  }

  #scheduleIdleTimer(idleTtlMs: number): void {
    this.#clearIdleTimer();
    if (idleTtlMs <= 0 || this.hasPendingToolTurn) return;
    this.#idleTimer = setTimeout(() => {
      if (!this.#busy && !this.hasPendingToolTurn) {
        void this.dispose();
      }
    }, idleTtlMs);
    this.#idleTimer.unref?.();
  }

  #clearIdleTimer(): void {
    if (this.#idleTimer) {
      clearTimeout(this.#idleTimer);
      this.#idleTimer = undefined;
    }
  }
}

/**
 * Pool of warm `AgyWorkerSlot` instances with sticky DSH session affinity and
 * isolated auxiliary-call execution (so title generation never disrupts main sessions).
 */
export class AgyWorkerPool {
  readonly #getConfig: () => ResolvedAgyAcpConfig;
  readonly #mcpBridge: LoopbackMcpBridge;
  readonly #catalog: AgyModelCatalog;
  readonly #loadImage?: ImageBytesLoader;

  readonly #slots: AgyWorkerSlot[] = [];
  #auxSlot?: AgyWorkerSlot;
  #slotSeq = 0;

  // Serialize turns targeting the exact same slot so concurrent requests queue cleanly
  readonly #slotLocks = new Map<AgyWorkerSlot, Promise<void>>();

  constructor(opts: {
    getConfig: () => ResolvedAgyAcpConfig;
    mcpBridge: LoopbackMcpBridge;
    catalog: AgyModelCatalog;
    loadImage?: ImageBytesLoader;
  }) {
    this.#getConfig = opts.getConfig;
    this.#mcpBridge = opts.mcpBridge;
    this.#catalog = opts.catalog;
    this.#loadImage = opts.loadImage;
  }

  async ensureCatalogLoaded(): Promise<void> {
    if (this.#catalog.snapshot) return;
    const slot = this.#acquireAuxSlot();
    await this.#withSlotLock(slot, async () => {
      await slot.probeCatalog();
    });
  }

  async *stream(options: GenerateOptions<string>): AsyncGenerator<StreamChunk> {
    const slot = this.#selectSlot(options);
    let releaseLock!: () => void;
    const prevLock = this.#slotLocks.get(slot) ?? Promise.resolve();
    const currentLock = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });
    this.#slotLocks.set(
      slot,
      prevLock.then(() => currentLock),
    );

    await prevLock;
    try {
      yield* slot.streamTurn(options);
    } finally {
      releaseLock();
    }
  }

  async dispose(): Promise<void> {
    const all = [...this.#slots, ...(this.#auxSlot ? [this.#auxSlot] : [])];
    this.#slots.length = 0;
    this.#auxSlot = undefined;
    this.#slotLocks.clear();
    await Promise.allSettled(all.map((s) => s.dispose()));
  }

  #acquireAuxSlot(): AgyWorkerSlot {
    if (!this.#auxSlot) {
      this.#auxSlot = new AgyWorkerSlot({
        id: 'aux',
        getConfig: this.#getConfig,
        mcpBridge: this.#mcpBridge,
        catalog: this.#catalog,
        loadImage: this.#loadImage,
      });
    }
    return this.#auxSlot;
  }

  #selectSlot(options: GenerateOptions<string>): AgyWorkerSlot {
    // Auxiliary model calls (e.g. session title generation) use dedicated `#auxSlot`
    if (options.purpose !== undefined) {
      return this.#acquireAuxSlot();
    }

    // 1. Match by active ACP session stored in ReplayEnvelope (handles fork/resume/tool-continuation)
    for (let i = options.messages.length - 1; i >= 0; i--) {
      const meta = extractReplayMetadata(options.messages[i]!);
      if (meta?.acpSessionId) {
        const byAcpId = this.#slots.find((s) => s.activeAcpSessionId === meta.acpSessionId);
        if (byAcpId) {
          if (options.sessionId) byAcpId.setBoundKey(String(options.sessionId));
          return byAcpId;
        }
        break;
      }
    }

    // 2. Match by DSH sessionId affinity
    const sessionKey = options.sessionId ? String(options.sessionId) : undefined;
    if (sessionKey) {
      const bound = this.#slots.find((s) => s.boundKey === sessionKey);
      if (bound) return bound;
    }

    // 3. Allocate new slot if pool has capacity
    const maxSlots = Math.max(1, this.#getConfig().poolSize);
    if (this.#slots.length < maxSlots) {
      const created = new AgyWorkerSlot({
        id: `worker-${++this.#slotSeq}`,
        getConfig: this.#getConfig,
        mcpBridge: this.#mcpBridge,
        catalog: this.#catalog,
        loadImage: this.#loadImage,
      });
      created.setBoundKey(sessionKey);
      this.#slots.push(created);
      return created;
    }

    // 4. Reuse least-recently-used idle slot without pending tool calls
    const candidates = this.#slots.filter((s) => !s.busy && !s.hasPendingToolTurn);
    const pool = candidates.length > 0 ? candidates : this.#slots;
    const chosen = pool.reduce((oldest, cur) =>
      cur.lastUsedAt < oldest.lastUsedAt ? cur : oldest,
    );
    chosen.setBoundKey(sessionKey);
    return chosen;
  }

  async #withSlotLock<T>(slot: AgyWorkerSlot, fn: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const prev = this.#slotLocks.get(slot) ?? Promise.resolve();
    const next = new Promise<void>((r) => {
      release = r;
    });
    this.#slotLocks.set(
      slot,
      prev.then(() => next),
    );
    await prev;
    try {
      return await fn();
    } finally {
      release();
    }
  }
}
