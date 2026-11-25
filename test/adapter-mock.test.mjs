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

const user = text => ({ role: 'user', content: [{ type: 'text', text }] });
function assistant(chunks) {
  return {
    id: 'reply', role: 'assistant',
    content: chunks.filter(c => c.type === 'block-end').map(c => c.block),
    source: { replayState: chunks.find(c => c.type === 'finish').replayState },
  };
}

function muxFixture(t, idleTtlMs = 0) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-mux-'));
  const cfg = { ...resolveAgyAcpConfig({ agyPath: FAKE_AGY_PATH, cacheDir: dir, workingDirectory: dir }), idleTtlMs };
  const adapter = new AgyAcpAdapter({ getConfig: () => cfg });
  t.after(async () => {
    await adapter.dispose();
    fs.rmSync(dir, { recursive: true, force: true });
  });
  return (sessionId, messages, tools = []) => collectChunks(adapter.stream({
    provider: 'agy-acp', model: 'gemini-4-carbon', sessionId, messages, tools,
  }));
}

test('eight interleaved sessions share one ACP child and isolate permissions, MCP routes, streams and results', { timeout: 12000 }, async t => {
  const turn = muxFixture(t);
  const members = Array.from({ length: 8 }, (_, i) => ({
    session: 'team-' + i,
    tools: ['run_code', 'member_' + i].map(name => ({ name, description: name, parameters: { type: 'object' } })),
    messages: [user('CALL_PTC_MEMBER_' + i)],
  }));
  const calls = await Promise.all(members.map(async m => assistant(await turn(m.session, m.messages, m.tools))));
  const routes = new Set(), pids = new Set(), sessions = new Set();
  await Promise.all(members.map(async (m, i) => {
    const reply = calls[i];
    sessions.add(reply.source.replayState.response.acpSessionId);
    const [call] = reply.content.filter(b => b.type === 'tool-call');
    assert.equal(call.name, 'run_code');
    const args = JSON.parse(call.arguments);
    assert.equal(args.member, String(i));
    assert.equal(args.own, 'allow_once');
    assert.equal(args.foreign, 'reject_once');
    assert.equal(args.unknown, 'reject_once');
    routes.add(args.route);
    pids.add(args.pid);
    const result = { role: 'tool', toolCallId: call.id, content: [{ type: 'text', text: 'reply-' + i }] };
    const final = assistant(await turn(m.session, [...m.messages, reply, result], m.tools));
    assert.deepEqual(final.content, [{ type: 'text', text: `member=${i} result=reply-${i} active=8` }]);
    assert.equal(final.source.replayState.response.acpSessionId, reply.source.replayState.response.acpSessionId);
  }));
  assert.equal(routes.size, 8);
  assert.equal(sessions.size, 8);
  assert.equal(pids.size, 1);
});

test('idle close releases just the idle session while a sibling MCP call waits for DSH', { timeout: 8000 }, async t => {
  const turn = muxFixture(t, 60);
  const tools = ['run_code', 'member_0'].map(name => ({ name, parameters: { type: 'object' } }));
  const start = user('CALL_PTC_MEMBER_0');
  const reply = assistant(await turn('waiting', [start], tools));
  await turn('idle', [user('hello')]);
  await new Promise(resolve => setTimeout(resolve, 240));
  const call = reply.content.find(b => b.type === 'tool-call');
  const final = assistant(await turn('waiting', [start, reply, {
    role: 'tool', toolCallId: call.id, content: [{ type: 'text', text: 'survived' }],
  }], tools));
  assert.equal(final.content[0].text, 'member=0 result=survived active=1');
});

test('process loss restores saved sessions, cold-replays unavailable sessions, and isolates forks', { timeout: 9000 }, async t => {
  const turn = muxFixture(t);
  const first = user('SAVE_FOR_LOAD');
  const saved = assistant(await turn('saved', [first]));
  const other = assistant(await turn('other', [user('hello')]));
  await assert.rejects(turn('crash', [user('EXIT_FIXTURE_PROCESS')]), /exited unexpectedly/);
  const [resumed, rebuilt, fork] = await Promise.all([
    turn('saved', [first, saved, user('continue')]),
    turn('other', [user('hello'), other, user('member report A'), user('member report B'), user('INSPECT_CURRENT_INPUT')]),
    turn('fork', [first, saved, user('forked')]),
  ].map(p => p.then(assistant)));
  const id = msg => msg.source.replayState.response.acpSessionId;
  assert.equal(id(resumed), id(saved));
  assert.notEqual(id(rebuilt), id(other));
  assert.match(rebuilt.content.at(-1).text, /member report A[\s\S]*member report B/);
  assert.doesNotMatch(rebuilt.content.at(-1).text, /hello/);
  assert.notEqual(id(fork), id(saved));
  assert.match(resumed.content.at(-1).text, /Turn=2/);
  assert.ok(resumed.content.every(b => !b.text.includes('historical load event')));
});
