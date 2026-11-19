import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgyAcpAdapter, resolveAgyAcpConfig } from '../dist/core.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FAKE_AGY_PATH = path.join(__dirname, 'fixtures', 'fake-agy.mjs');
fs.chmodSync(FAKE_AGY_PATH, 0o755);

async function collectChunks(iterable) {
  const chunks = [];
  for await (const chunk of iterable) {
    chunks.push(chunk);
  }
  return chunks;
}

test('AgyAcpAdapter streams reasoning, parallel tool-calls, warm continuation, images, and isolates aux calls', async () => {
  const tmpCache = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-adapter-test-'));
  const cfg = resolveAgyAcpConfig({
    agyPath: FAKE_AGY_PATH,
    poolSize: 2,
    batchWindowMs: 30,
    cacheDir: tmpCache,
  });

  const adapter = new AgyAcpAdapter({
    getConfig: () => cfg,
    loadImage: async (ref) => ({
      base64: Buffer.from(`PIXELS:${ref.attachmentId}`).toString('base64'),
      mimeType: ref.mediaType,
    }),
  });

  try {
    const models = await adapter.listModels('agy-acp');
    assert.equal(models.length, 2);
    assert.equal(models[0].id, 'gemini-4-carbon');
    assert.equal(models[1].id, 'gemini-4-hydrogen');

    const resolvedHydrogen = await adapter.resolveModel('agy-acp', 'gemini-4-hydrogen');
    assert.deepEqual(
      resolvedHydrogen.reasoning?.efforts.map((e) => e.id),
      ['high', 'xhigh'],
    );

    const tools = [
      { name: 'get_left', description: 'Left', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
      { name: 'get_right', description: 'Right', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
    ];

    // Step 1: User triggers parallel tool calls
    const user1 = {
      id: 'u1',
      role: 'user',
      content: [{ type: 'text', text: 'Please CALL_PARALLEL_TOOLS now' }],
      source: { kind: 'user' },
    };

    const step1Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        reasoningEffort: 'high',
        sessionId: 'dsh-sess-1',
        messages: [user1],
        tools,
      }),
    );

    const step1Finish = step1Chunks.find((c) => c.type === 'finish');
    assert.equal(step1Finish?.reason.kind, 'tool-calls');

    const toolBlocks = step1Chunks
      .filter((c) => c.type === 'block-end' && c.block.type === 'tool-call')
      .map((c) => c.block);
    assert.equal(toolBlocks.length, 2, 'Both parallel tools coalesced into single DSH turn');
    assert.equal(toolBlocks[0].arguments, '{"q":"L"}', 'toolSummary stripped cleanly');
    assert.equal(toolBlocks[1].arguments, '{"q":"R"}', 'toolSummary stripped cleanly');

    const reasoningEnd = step1Chunks.find(
      (c) => c.type === 'block-end' && c.block.type === 'reasoning',
    );
    assert.equal(reasoningEnd?.block.text, 'Thinking step...');

    // Step 2: DSH returns both ToolResultMessages -> warm worker resolves MCP calls & finishes
    const asstToolMsg = {
      id: 'a1',
      role: 'assistant',
      content: toolBlocks,
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: step1Finish.replayState,
      },
    };
    const toolRes1 = {
      id: 't1',
      role: 'tool',
      toolCallId: toolBlocks[0].id,
      content: [{ type: 'text', text: 'ALPHA' }],
      source: { kind: 'tool' },
    };
    const toolRes2 = {
      id: 't2',
      role: 'tool',
      toolCallId: toolBlocks[1].id,
      content: [{ type: 'text', text: 'BETA' }],
      source: { kind: 'tool' },
    };

    const step2Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        reasoningEffort: 'high',
        sessionId: 'dsh-sess-1',
        messages: [user1, asstToolMsg, toolRes1, toolRes2],
        tools,
      }),
    );

    const step2Text = step2Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
    assert.equal(step2Text?.block.text, 'Combined=ALPHA+BETA');

    const step2Usage = step2Chunks.find((c) => c.type === 'usage');
    assert.equal(step2Usage?.usage.inputTokens, 1000);
    assert.equal(step2Usage?.usage.cacheReadTokens, 400);
    assert.equal(step2Usage?.usage.reasoningTokens, 15);
    assert.equal(step2Usage?.usage.totalTokens, 1450);

    const step2Finish = step2Chunks.find((c) => c.type === 'finish');
    assert.equal(step2Finish?.reason.kind, 'stop');
    const firstSessionId = step2Finish.replayState.response.acpSessionId;

    // Auxiliary call (e.g. session-title) must NOT disturb `firstSessionId`!
    const auxChunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        purpose: 'session-title',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'Generate title' }] }],
      }),
    );
    const auxFinish = auxChunks.find((c) => c.type === 'finish');
    assert.notEqual(auxFinish.replayState.response.acpSessionId, firstSessionId);

    // Turn 2 on `dsh-sess-1` with multimodal ImageBlock reuses warm `firstSessionId` (Turn=2)!
    const asstFinal1 = {
      id: 'a2',
      role: 'assistant',
      content: [step2Text.block],
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: step2Finish.replayState,
      },
    };
    const user2WithImage = {
      id: 'u2',
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect this screenshot' },
        {
          type: 'image',
          attachment: {
            attachmentId: 'img-99',
            mediaType: 'image/png',
            bytes: 128,
            width: 100,
            height: 80,
          },
        },
      ],
      source: { kind: 'user' },
    };

    const turn2Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-hydrogen',
        reasoningEffort: 'xhigh',
        sessionId: 'dsh-sess-1',
        messages: [user1, asstToolMsg, toolRes1, toolRes2, asstFinal1, user2WithImage],
        tools,
      }),
    );

    const turn2Text = turn2Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
    assert.match(
      turn2Text.block.text,
      new RegExp(`Session=${firstSessionId} Turn=2 Model=gemini-4-hydrogen-xhigh \\[img:image/png:`),
    );

    // 4. Verify upstream 429 circuit breaker trips immediately on attempt 1, cancels agy, and throws non-retryable QUOTA error
    const leakedText = [];
    await assert.rejects(
      async () => {
        for await (const chunk of adapter.stream({
          provider: 'agy-acp',
          model: 'gemini-4-carbon',
          system: 'You are an AI agent powered by DeepSeek Harness.\nFollow workspace rules.',
          messages: [
            {
              id: 'u-429',
              role: 'user',
              content: [{ type: 'text', text: 'SIMULATE_UPSTREAM_429' }],
              source: { kind: 'user' },
            },
          ],
        })) {
          if (chunk.type === 'text-delta') leakedText.push(chunk.text);
        }
      },
      (err) => {
        assert.equal(err.code, 'QUOTA');
        assert.match(err.message, /circuit breaker tripped on attempt 1/);
        return true;
      },
    );
    assert.deepEqual(leakedText, [], 'In-band 429 banner must never leak as assistant text');
  } finally {
    await adapter.dispose();
    fs.rmSync(tmpCache, { recursive: true, force: true });
  }
});
