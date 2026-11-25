import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  AgyModelCatalog,
  LoopbackMcpBridge,
  McpSessionEndpoint,
  buildCatalogEntries,
  buildFullReplayPromptParts,
  classifyInBandAgyApiError,
  computeContextDigests,
  computeToolsFingerprint,
  createSanitizedAgyEnv,
  encodeMcpToolName,
  formatHarnessGuidelines,
  planSessionSync,
  resolveAgyBinary,
  sanitizeMcpToolArguments,
  truncateToolOutputUtf8,
} from '../dist/core.js';

test('createSanitizedAgyEnv strips ambient API keys to preserve subscription OAuth', () => {
  const dirty = {
    PATH: '/usr/bin',
    HOME: '/Users/demo',
    GEMINI_API_KEY: 'AIzaSyFakeKey',
    GOOGLE_API_KEY: 'AIzaSyFakeKey2',
    AGY_ADC_AUTH: '1',
    DSH_AGY_LOCAL_KEY: 'secret',
  };
  const clean = createSanitizedAgyEnv(dirty);
  assert.equal(clean.PATH, '/usr/bin');
  assert.equal(clean.GEMINI_API_KEY, undefined);
  assert.equal(clean.GOOGLE_API_KEY, undefined);
  assert.equal(clean.AGY_ADC_AUTH, undefined);
  assert.equal(clean.DSH_AGY_LOCAL_KEY, undefined);
});

test('resolveAgyBinary validates explicit path and rejects missing binary', () => {
  assert.throws(
    () => resolveAgyBinary('/nonexistent/path/to/agy'),
    /not found or is not executable/,
  );
});

test('AgyModelCatalog groups families, supports Hydrogen xhigh, and persists cache', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-cat-test-'));
  try {
    const rawModels = [
      { modelId: 'gemini-4-carbon-medium', name: 'Gemini 4 Carbon (Medium)' },
      { modelId: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)' },
      { modelId: 'gemini-4-carbon-low', name: 'Gemini 4 Carbon (Low)' },
      { modelId: 'gemini-4-hydrogen-high', name: 'Gemini 4 Hydrogen (High)' },
      { modelId: 'gemini-4-hydrogen-xhigh', name: 'Gemini 4 Hydrogen (XHigh)' },
      { modelId: 'gemini-4-hydrogen-medium', name: 'Gemini 4 Hydrogen (Medium)' },
      { modelId: 'gemini-4-hydrogen-low', name: 'Gemini 4 Hydrogen (Low)' },
      { modelId: 'gemini-pro-orchestrator', name: 'Orchestrator (Carbon)' },
      { modelId: 'gpt-oss-120b-medium', name: 'GPT-OSS 120B (Medium)' },
    ];

    const entries = buildCatalogEntries(rawModels);
    assert.equal(entries.length, 4);

    const hydrogen = entries.find((e) => e.id === 'gemini-4-hydrogen');
    assert.ok(hydrogen);
    assert.equal(hydrogen.name, 'Gemini 4 Hydrogen');
    assert.deepEqual([...hydrogen.effortToRawModelId.keys()], ['low', 'medium', 'high', 'xhigh']);
    assert.equal(hydrogen.defaultEffort, 'high');

    const catalog = new AgyModelCatalog(tmpDir);
    catalog.update(rawModels, { name: 'Google AI Ultra', id: 'ultra' });

    assert.equal(catalog.resolveRawModelId('gemini-4-hydrogen', 'xhigh'), 'gemini-4-hydrogen-xhigh');
    assert.equal(catalog.resolveRawModelId('gemini-4-carbon'), 'gemini-4-carbon-medium');
    assert.equal(catalog.resolveRawModelId('gemini-4-carbon-high'), 'gemini-4-carbon-high');
    assert.equal(catalog.resolveRawModelId('gemini-pro-orchestrator'), 'gemini-pro-orchestrator');
    assert.equal(catalog.resolveRawModelId('gpt-oss-120b-medium'), 'gpt-oss-120b-medium');

    const reloaded = new AgyModelCatalog(tmpDir);
    assert.equal(reloaded.tier?.name, 'Google AI Ultra');
    const infos = reloaded.toLlmModelInfos('agy-acp');
    assert.equal(infos.length, 4);
    assert.deepEqual(infos[0].inputModalities, ['text', 'image']);
    assert.equal(infos[0].badge?.text, 'Ultra');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('sanitizeMcpToolArguments strips undeclared agy toolSummary but keeps declared one', () => {
  const strictSchema = {
    name: 'workspace_read',
    description: 'Read file',
    parameters: {
      type: 'object',
      properties: { file_path: { type: 'string' } },
      additionalProperties: false,
    },
  };
  assert.deepEqual(
    sanitizeMcpToolArguments(
      { file_path: '/tmp/a.txt', toolSummary: 'Read /tmp/a.txt' },
      strictSchema,
    ),
    { file_path: '/tmp/a.txt' },
  );

  const schemaWithSummary = {
    name: 'custom_summary_tool',
    description: 'Has toolSummary',
    parameters: {
      type: 'object',
      properties: { file_path: { type: 'string' }, toolSummary: { type: 'string' } },
    },
  };
  assert.deepEqual(
    sanitizeMcpToolArguments(
      { file_path: '/tmp/a.txt', toolSummary: 'Keep me' },
      schemaWithSummary,
    ),
    { file_path: '/tmp/a.txt', toolSummary: 'Keep me' },
  );
});

test('truncateToolOutputUtf8 never splits multi-byte CJK characters', () => {
  const input = '你好世界🚀'.repeat(200);
  const truncated = truncateToolOutputUtf8(input, 300);
  assert.ok(Buffer.byteLength(truncated, 'utf8') <= 300);
  assert.ok(!truncated.includes('\uFFFD'), 'Should not contain UTF-8 replacement char');
  assert.match(truncated, /Output truncated at 300 bytes/);
});

test('LoopbackMcpBridge enforces token auth and coalesces parallel tool calls into one batch', async () => {
  const bridge = new LoopbackMcpBridge();
  try {
    const endpoint = await bridge.createSessionEndpoint({
      maxToolOutputBytes: 16384,
      batchWindowMs: 30,
    });
    endpoint.setTools([
      {
        name: 'read_a',
        description: 'Read A',
        parameters: { type: 'object', properties: { id: { type: 'string' } } },
      },
      {
        name: 'read_b',
        description: 'Read B',
        parameters: { type: 'object', properties: { id: { type: 'string' } } },
      },
    ]);

    const cfg = endpoint.toAcpMcpServerConfig();

    // Unauthorized request must fail with 401
    const unauth = await fetch(cfg.url, {
      method: 'POST',
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }),
    });
    assert.equal(unauth.status, 401);

    const sibling = await bridge.createSessionEndpoint({ maxToolOutputBytes: 16384, batchWindowMs: 0 });
    const wrongRoute = await fetch(sibling.toAcpMcpServerConfig().url, {
      method: 'POST', headers: Object.fromEntries(cfg.headers.map(h => [h.name, h.value])),
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }),
    });
    assert.equal(wrongRoute.status, 401);
    sibling.dispose();

    // Send two concurrent tools/call requests within batchWindowMs
    const batches = [];
    endpoint.onNextBatch((batch) => batches.push(batch));

    const headers = Object.fromEntries(cfg.headers.map((h) => [h.name, h.value]));
    const callA = fetch(cfg.url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 10,
        method: 'tools/call',
        params: { name: 'read_a', arguments: { id: '1', toolSummary: 'summary a' } },
      }),
    });
    const callB = fetch(cfg.url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 11,
        method: 'tools/call',
        params: { name: 'read_b', arguments: { id: '2', toolSummary: 'summary b' } },
      }),
    });

    await new Promise((r) => setTimeout(r, 60));
    assert.equal(batches.length, 1, 'Expected both calls coalesced into a single batch');
    assert.equal(batches[0].length, 2);
    assert.equal(batches[0][0].argumentsJson, '{"id":"1"}');
    assert.equal(batches[0][1].argumentsJson, '{"id":"2"}');

    endpoint.resolveToolCall(batches[0][0].toolCallId, { text: 'RES_A' });
    endpoint.resolveToolCall(batches[0][1].toolCallId, { text: 'RES_B', isError: true });

    const [resA, resB] = await Promise.all([callA.then((r) => r.json()), callB.then((r) => r.json())]);
    assert.equal(resA.result.content[0].text, 'RES_A');
    assert.equal(resB.result.content[0].text, 'RES_B');
    assert.equal(resB.result.isError, true);
  } finally {
    await bridge.close();
  }
});

test('planSessionSync tolerates Context OS surfaceOp: replace and runtime notices without resetting session', async () => {
  const toolsFp = computeToolsFingerprint([{ name: 'bash', description: 'Run', parameters: {} }]);

  const initialMessages = [
    {
      id: 'dev-surface-1',
      role: 'developer',
      content: [{ type: 'text', text: 'Todo status: step 1 pending' }],
      source: { kind: 'developer' },
    },
    {
      id: 'user-1',
      role: 'user',
      content: [{ type: 'text', text: 'Inspect project' }],
      source: { kind: 'user' },
    },
  ];
  const initialDigests = computeContextDigests(initialMessages);

  // Assistant emitted a tool call; Context OS then replaced `dev-surface-1` AND DSH added runtime notice
  const midTurnMessages = [
    {
      id: 'dev-surface-1',
      role: 'developer',
      content: [{ type: 'text', text: 'Todo status: step 1 IN PROGRESS (mutated by Context OS)' }],
      source: { kind: 'developer' },
    },
    initialMessages[1],
    {
      id: 'asst-1',
      role: 'assistant',
      content: [{ type: 'tool-call', id: 'call_1', name: 'bash', arguments: '{"cmd":"ls"}' }],
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: {
          response: {
            version: 1,
            acpSessionId: 'sess-live-1',
            rawModelId: 'gemini-4-carbon-medium',
            syncedMessageCount: 3,
            toolsFingerprint: toolsFp,
            contextDigests: initialDigests,
          },
        },
      },
    },
    {
      id: 'tool-res-1',
      role: 'tool',
      toolCallId: 'call_1',
      content: [{ type: 'text', text: 'README.md\nsrc/' }],
      source: { kind: 'tool' },
    },
    {
      id: 'runtime-note-1',
      role: 'user',
      content: [{ type: 'text', text: 'Time sampled while preparing turn 2, step 3: 2026-11-18T17:00:00+08:00' }],
      source: { kind: 'user' },
    },
  ];

  const midTurnDecision = planSessionSync({
    messages: midTurnMessages,
    activeAcpSessionId: 'sess-live-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastContextDigests: initialDigests,
    pendingToolCallIds: new Set(['call_1']),
    allowResumeFromReplay: true,
    forwardContextUpdates: true,
  });

  assert.equal(midTurnDecision.kind, 'resolve-pending-tools');
  assert.equal(midTurnDecision.resolutions.length, 1);
  assert.equal(midTurnDecision.resolutions[0].text, 'README.md\nsrc/');

  // Next turn: assistant finished, Context OS mutated `dev-surface-1` again, user sends Turn 2
  const turn2Messages = [
    {
      id: 'dev-surface-1',
      role: 'developer',
      content: [{ type: 'text', text: 'Todo status: step 1 COMPLETED, step 2 IN PROGRESS' }],
      source: { kind: 'developer' },
    },
    initialMessages[1],
    midTurnMessages[2],
    midTurnMessages[3],
    {
      id: 'asst-final-1',
      role: 'assistant',
      content: [{ type: 'text', text: 'Found README.md and src/.' }],
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: {
          response: {
            version: 1,
            acpSessionId: 'sess-live-1',
            rawModelId: 'gemini-4-carbon-medium',
            syncedMessageCount: 5,
            toolsFingerprint: toolsFp,
            contextDigests: initialDigests,
          },
        },
      },
    },
    {
      id: 'user-2',
      role: 'user',
      content: [{ type: 'text', text: 'Now read README.md' }],
      source: { kind: 'user' },
    },
  ];

  const turn2Decision = planSessionSync({
    messages: turn2Messages,
    activeAcpSessionId: 'sess-live-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastContextDigests: initialDigests,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
    forwardContextUpdates: true,
  });

  assert.equal(turn2Decision.kind, 'warm-incremental');
  assert.equal(turn2Decision.tailMessages.length, 1);
  assert.equal(turn2Decision.updatedContextBlocks.length, 1);
  assert.match(turn2Decision.updatedContextBlocks[0], /step 2 IN PROGRESS/);

  // Cold replay preserves structured tool calls, tool results, and images
  const replayParts = await buildFullReplayPromptParts({
    messages: turn2Messages,
  });
  const combinedReplayText = replayParts.map((p) => (p.type === 'text' ? p.text : '')).join('\n');
  assert.match(combinedReplayText, /<tool_call id="call_1" name="bash">/);
  assert.match(combinedReplayText, /<tool_result id="call_1" status="ok">/);
  assert.match(combinedReplayText, /Now read README\.md/);
});

test('formatHarnessGuidelines & buildFullReplayPromptParts strip third-party persona headers and place runtime reminders before human request', async () => {
  const rawSystem = [
    'You are an AI agent powered by DeepSeek Harness.',
    '',
    'You are a coding agent powered by the gemini-3.8-flash model.',
    '',
    'Use the read tool — not shell commands like cat — to inspect text files.',
  ].join('\n');

  const guidelines = formatHarnessGuidelines(rawSystem, true);
  assert.ok(guidelines);
  assert.doesNotMatch(guidelines, /powered by DeepSeek Harness/i);
  assert.doesNotMatch(guidelines, /You are a coding agent/i);
  assert.match(guidelines, /mcp_dsh_<tool_name>/);
  assert.match(guidelines, /Use the read tool/);

  const parts = await buildFullReplayPromptParts({
    systemPrompt: rawSystem,
    hasTools: true,
    messages: [
      {
        id: 'human-1',
        role: 'user',
        content: [{ type: 'text', text: 'Reply with PONG_OK only.' }],
        source: { kind: 'user' },
      },
      {
        role: 'user',
        content: [{ type: 'text', text: '<system-reminder>\nWorkspace skills catalog...\n</system-reminder>' }],
      },
      {
        role: 'user',
        content: [{ type: 'text', text: 'Current runtime context. This snapshot supersedes earlier snapshots.' }],
      },
    ],
  });

  const fullText = parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n');
  const reminderPos = fullText.indexOf('<system-reminder>');
  const humanPos = fullText.indexOf('Reply with PONG_OK only.');
  assert.ok(reminderPos > 0);
  assert.ok(humanPos > reminderPos, 'Human request must appear after trailing runtime reminders');
});

test('workspace guidelines match the registered bridge names, including PTC-only and empty catalogs', async () => {
  const endpoint = new McpSessionEndpoint({
    token: 'fixture', url: 'http://localhost', maxToolOutputBytes: 1024, batchWindowMs: 0, onDispose() {},
  });
  const setTools = names => endpoint.setTools(names.map(name => ({
    name, description: name, parameters: { type: 'object' },
  })));
  const prompt = async () => (await buildFullReplayPromptParts({
    messages: [], bridgeToolNames: endpoint.bridgeToolNames,
  }))[0].text;

  setTools(['issue/search', 'issue_search']);
  const standard = await prompt();
  assert.equal(new Set(endpoint.bridgeToolNames).size, 2);
  for (const name of endpoint.bridgeToolNames) assert.ok(standard.includes(`mcp_dsh_${name}`));
  assert.doesNotMatch(standard, /mcp_dsh_read|mcp_dsh_run_code/);
  setTools(['run_code']);
  const ptc = await prompt();
  assert.match(ptc, /mcp_dsh_run_code/);
  assert.match(ptc, /exact DSH tool names in the supplied SDK/);
  assert.match(ptc, /ambient MCP servers/);
  assert.doesNotMatch(ptc, /issue_search/);
  setTools([]);
  assert.match(await prompt(), /No DSH tools/);
});

test('an existing literal MCP name is preserved when an encoded collision needs the same suffix', () => {
  const original = 'issue.search';
  const base = 'issue_search';
  const literal = encodeMcpToolName(original, new Set([base]));
  const used = new Set([base, literal]);
  const encoded = encodeMcpToolName(original, used);
  assert.equal(used.size, 3);
  assert.ok(used.has(encoded));
  assert.notEqual(encoded, literal);
});

for (const [preset, toolNames] of Object.entries({
  standard: ['read', 'mcp__sample-server__list_items'],
  ptc: ['run_code'],
  minimal: ['read', 'bash'],
  cordis: ['cordis', 'read', 'plugin_manager'],
  empty: [],
  custom: ['evaluate.workflow', 'evaluate_workflow', 'mcp__x-y__fetch', 'a'.repeat(70) + 'x', 'a'.repeat(70) + 'y'],
})) {
  test(`${preset} routes exactly the tools projected by the request`, () => {
    const endpoint = new McpSessionEndpoint({
      token: 'fixture', url: 'http://localhost', maxToolOutputBytes: 1024, batchWindowMs: 0, onDispose() {},
    });
    endpoint.setTools(toolNames.map(name => ({ name, description: name, parameters: { type: 'object' } })));
    const names = endpoint.bridgeToolNames;
    assert.equal(new Set(names).size, toolNames.length);
    const guidelines = formatHarnessGuidelines(undefined, names);
    const advertised = guidelines.match(/Direct calls available in this request: (.*?)\./);
    assert.deepEqual(advertised ? advertised[1].split(', ') : [], names.map(name => `mcp_dsh_${name}`));
    for (const name of names) {
      assert.match(name, /^[a-zA-Z0-9_-]{1,64}$/);
      assert.equal(endpoint.isAllowedPermissionRequest(`mcp_dsh_${name}`), true);
      assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'dsh', ToolName: name }), true);
      assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'external', ToolName: name }), false);
    }
    endpoint.setTools([]);
    for (const name of names) assert.equal(endpoint.isAllowedPermissionRequest(`mcp_dsh_${name}`), false);
  });
}

test('ACP allows active eager and dsh dispatcher calls and rejects other requests', () => {
  const endpoint = new McpSessionEndpoint({
    token: 'fixture', url: 'http://localhost', maxToolOutputBytes: 1024, batchWindowMs: 0, onDispose() {},
  });
  endpoint.setTools([{ name: 'run_code', parameters: { type: 'object' }, description: 'PTC' }]);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_run_code', { toolSummary: 'Run program' }), true);
  assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'dsh', ToolName: 'run_code' }), true);
  assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'external', ToolName: 'run_code' }), false);
  assert.equal(endpoint.isAllowedPermissionRequest('run_code'), false);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_unknown'), false);
  endpoint.setTools([]);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_run_code'), false);
});

test('classifyInBandAgyApiError detects standalone agy 429/403 retry banners without false positives in prose', () => {
  const err429 = classifyInBandAgyApiError(
    '\n⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).\n',
  );
  assert.ok(err429);
  assert.equal(err429.code, 'QUOTA');

  const errAuth = classifyInBandAgyApiError(
    '\n⚠ API error (attempt 1): UNAUTHENTICATED (code 401): Token expired.\n',
  );
  assert.ok(errAuth);
  assert.equal(errAuth.code, 'INVALID_CREDENTIAL');

  // Prose discussion mentioning the warning in the middle of text must NOT trigger
  const prose = classifyInBandAgyApiError(
    'Earlier we saw `⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): ...` in the logs.',
  );
  assert.equal(prose, undefined);
});

test('cold replay loads only current images and keeps consecutive teammate reports after the last tool result', async () => {
  const text = value => ({ type: 'text', text: value });
  const image = id => ({ type: 'image', attachment: { attachmentId: id, width: 10, height: 20 } });
  const read = [];
  const parts = await buildFullReplayPromptParts({
    messages: [
      { role: 'user', content: [image('old.png')] },
      { role: 'assistant', content: [text('Checking')] },
      { role: 'tool', toolCallId: 'a', content: [text('Checked'), image('tool.png')] },
      { role: 'user', content: [text('Programmer report')] },
      { role: 'developer', content: [text('Current team context')] },
      { role: 'user', content: [text('Reviewer report'), image('current.png')] },
      { role: 'user', content: [text('Coordinator report')] },
    ],
    loadImage: async ref => {
      read.push(ref.attachmentId);
      return { mimeType: 'image/png', base64: 'eA==' };
    },
  });
  assert.deepEqual(read, ['current.png']);
  assert.equal(parts.filter(p => p.type === 'image').length, 1);
  const [history, current] = parts.map(p => p.text ?? '[pixels]').join('\n').split('<current_input>');
  assert.match(history, /old\.png/);
  assert.match(history, /tool\.png/);
  assert.doesNotMatch(history, /report|pixels/);
  assert.match(current, /Programmer report[\s\S]*Reviewer report[\s\S]*\[pixels\][\s\S]*Coordinator report/);
  assert.equal(current.match(/<user_message>/g)?.length, 3);
});
