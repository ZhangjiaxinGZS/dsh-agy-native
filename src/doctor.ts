import { ReasoningEffortId } from '@deepseek-ai/dsh-llm';
import { AcpWorkerClient } from './acp-client.js';
import { checkAgyStateDirectory, probeAgyVersion, resolveAgyBinary } from './binary.js';
import { buildCatalogEntries } from './catalog.js';
import type { AgyDoctorReport } from './types.js';

/**
 * Perform a non-destructive end-to-end health check against the local official `agy` binary:
 * binary lookup -> state directory writability -> ACP handshake -> `session/new` tier & models.
 * Consumes zero model tokens.
 */
export async function runAgyDoctor(opts?: {
  agyPath?: string;
  startupTimeoutMs?: number;
}): Promise<AgyDoctorReport> {
  const { stateDir, writable: stateDirWritable } = checkAgyStateDirectory();

  let agyPath: string;
  try {
    agyPath = resolveAgyBinary(opts?.agyPath);
  } catch (err) {
    return {
      ok: false,
      stateDir,
      stateDirWritable,
      rawModelCount: 0,
      familyModels: [],
      error: err instanceof Error ? err.message : String(err),
    };
  }

  const agyVersion = await probeAgyVersion(agyPath);

  if (!stateDirWritable) {
    return {
      ok: false,
      agyPath,
      agyVersion,
      stateDir,
      stateDirWritable,
      rawModelCount: 0,
      familyModels: [],
      error: `Antigravity state directory is not writable: ${stateDir}`,
    };
  }

  const client = new AcpWorkerClient({
    agyPath,
    startupTimeoutMs: opts?.startupTimeoutMs ?? 20_000,
  });

  try {
    const caps = await client.start();
    const session = await client.createSession({});
    await client.closeSession(session.sessionId);

    const entries = buildCatalogEntries(session.availableModels);
    const familyModels = entries.map((e) => ({
      id: e.id,
      name: e.name,
      efforts: [...e.effortToRawModelId.keys()].map((k) => ReasoningEffortId(k)),
      ...(e.defaultEffort ? { defaultEffort: ReasoningEffortId(e.defaultEffort) } : {}),
    }));

    return {
      ok: session.availableModels.length > 0,
      agyPath,
      agyVersion,
      stateDir,
      stateDirWritable,
      protocolVersion: caps.protocolVersion,
      capabilities: {
        image: caps.image,
        loadSession: caps.loadSession,
        systemPromptAppend: caps.systemPromptAppend,
      },
      tier: session.tier,
      rawModelCount: session.availableModels.length,
      familyModels,
      ...(session.availableModels.length === 0
        ? {
            error:
              'No models returned by `agy acp`. Run `agy login` in your terminal to sign in to your Google AI subscription.',
          }
        : {}),
    };
  } catch (err) {
    return {
      ok: false,
      agyPath,
      agyVersion,
      stateDir,
      stateDirWritable,
      rawModelCount: 0,
      familyModels: [],
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    await client.dispose();
  }
}
