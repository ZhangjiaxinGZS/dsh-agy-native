#!/usr/bin/env node
import { runAgyDoctor } from './doctor.js';

function printHelp(): void {
  console.log(`dsh-agy-native — TOS-safe Antigravity ACP provider for DeepSeek Harness

Usage:
  dsh-agy-native doctor [--json] [--agy-path <path>]
  dsh-agy-native --help

Commands:
  doctor    Verify local official 'agy' CLI installation, subscription tier,
            ACP capabilities, and discovered model families (uses zero tokens).
`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes('--help') || args.includes('-h')) {
    printHelp();
    return;
  }

  const command = args[0] && !args[0].startsWith('-') ? args[0] : 'doctor';
  if (command !== 'doctor') {
    console.error(`Unknown command: ${command}`);
    printHelp();
    process.exitCode = 1;
    return;
  }

  const jsonMode = args.includes('--json');
  const pathFlagIdx = args.indexOf('--agy-path');
  const agyPath = pathFlagIdx >= 0 ? args[pathFlagIdx + 1] : undefined;

  const report = await runAgyDoctor({ agyPath });
  if (jsonMode) {
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) process.exitCode = 1;
    return;
  }

  console.log('=== dsh-agy-native Doctor ===');
  console.log(`Status:             ${report.ok ? 'OK' : 'FAILED'}`);
  console.log(`Official agy path:  ${report.agyPath ?? '(not found)'}`);
  if (report.agyVersion) {
    console.log(`agy version:        ${report.agyVersion}`);
  }
  console.log(
    `State directory:    ${report.stateDir} (${report.stateDirWritable ? 'writable' : 'NOT WRITABLE'})`,
  );
  if (report.tier?.name) {
    console.log(`Subscription tier:  ${report.tier.name} (${report.tier.id ?? 'n/a'})`);
  }
  if (report.capabilities) {
    console.log(
      `ACP capabilities:   v${report.protocolVersion} | image=${report.capabilities.image} | loadSession=${report.capabilities.loadSession} | systemPrompt.append=${report.capabilities.systemPromptAppend}`,
    );
  }
  if (report.familyModels.length > 0) {
    console.log(
      `\nDiscovered ${report.familyModels.length} model routes (${report.rawModelCount} raw variants):`,
    );
    for (const m of report.familyModels) {
      const efforts =
        m.efforts.length > 0
          ? ` [efforts: ${m.efforts.join(', ')}; default=${m.defaultEffort}]`
          : '';
      console.log(`  - ${m.id} (${m.name})${efforts}`);
    }
  }
  if (report.error) {
    console.error(`\nError: ${report.error}`);
    process.exitCode = 1;
  }
}

void main();
