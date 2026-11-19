import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Context } from '@deepseek-ai/cordis';
import LlmRuntime, { ReasoningEffortId } from '@deepseek-ai/dsh-llm';
import AgyAcpPlugin from '../dist/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FAKE_AGY_PATH = path.join(__dirname, 'fixtures', 'fake-agy.mjs');

test('dsh-agy-native mounts cleanly in real Cordis + @deepseek-ai/dsh-llm LlmRuntime and unloads cleanly', async (t) => {
  const tmpCache = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-cordis-test-'));
  const ctx = new Context();
  t.after(async () => {
    await ctx.fiber.dispose();
    fs.rmSync(tmpCache, { recursive: true, force: true });
  });

  const llmFiber = ctx.plugin(LlmRuntime);
  await llmFiber.inertia;

  const pluginFiber = ctx.plugin(AgyAcpPlugin, {
    agyPath: FAKE_AGY_PATH,
    cacheDir: tmpCache,
  });
  await pluginFiber.inertia;

  const providers = ctx.llm.listProviders();
  assert.equal(providers.length, 1);
  assert.equal(providers[0].id, 'agy-acp');

  const configurable = ctx.llm.listConfigurableProviders();
  assert.equal(configurable.length, 1);
  assert.equal(configurable[0].provider, 'agy-acp');
  assert.equal(configurable[0].nativeAuth?.oauth, true);

  const models = await ctx.llm.listModels('agy-acp');
  assert.equal(models.length, 2);
  assert.equal(models[0].id, 'gemini-4-carbon');
  assert.equal(models[1].id, 'gemini-4-hydrogen');

  // After discovery, provider name reflects discovered subscription tier
  const updatedProviders = ctx.llm.listProviders();
  assert.equal(updatedProviders[0].name, 'Antigravity (Google AI Ultra)');

  const prepared = await ctx.llm.prepareCall({
    provider: 'agy-acp',
    model: 'gemini-4-hydrogen',
    reasoningEffort: ReasoningEffortId('xhigh'),
  });
  assert.equal(prepared.config.model, 'gemini-4-hydrogen');
  assert.equal(prepared.config.reasoningEffort, 'xhigh');

  const chunks = [];
  for await (const chunk of prepared.stream({
    ...prepared.config,
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Hello via LlmRuntime' }] }],
  })) {
    chunks.push(chunk);
  }

  const textEnd = chunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
  assert.match(textEnd.block.text, /Model=gemini-4-hydrogen-xhigh/);

  // Unload plugin fiber and verify provider route is cleanly released
  await pluginFiber.dispose();
  assert.equal(ctx.llm.listProviders().length, 0);
  assert.equal(ctx.llm.listConfigurableProviders().length, 0);
});
