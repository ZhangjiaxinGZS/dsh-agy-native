import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

/**
 * Environment variables stripped before spawning `agy` so it never silently
 * falls back from Google AI Pro/Ultra subscription OAuth to pay-per-token API keys.
 */
export const STRIPPED_AUTH_ENV_KEYS = [
  'GEMINI_API_KEY',
  'GOOGLE_API_KEY',
  'AGY_ADC_AUTH',
  'DSH_AGY_LOCAL_KEY',
] as const;

function expandTilde(input: string): string {
  if (input === '~') return os.homedir();
  if (input.startsWith('~/') || input.startsWith('~\\')) {
    return path.join(os.homedir(), input.slice(2));
  }
  return input;
}

function isExecutableFile(candidate: string): boolean {
  try {
    const stat = fs.statSync(candidate);
    if (!stat.isFile()) return false;
    if (process.platform === 'win32') return true;
    fs.accessSync(candidate, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function searchPathForBinary(names: readonly string[], envPath = process.env.PATH ?? ''): string | undefined {
  const dirs = envPath.split(path.delimiter).filter(Boolean);
  for (const dir of dirs) {
    for (const name of names) {
      const candidate = path.join(dir, name);
      if (isExecutableFile(candidate)) return candidate;
    }
  }
  return undefined;
}

/**
 * Discover the official local `agy` executable without invoking any unofficial endpoints.
 */
export function resolveAgyBinary(explicitPath?: string, env: NodeJS.ProcessEnv = process.env): string {
  const configured = explicitPath?.trim() || env.AGY_BIN?.trim();
  if (configured) {
    const resolved = path.resolve(expandTilde(configured));
    if (!isExecutableFile(resolved)) {
      throw new Error(
        `dsh-agy-native: configured Antigravity CLI binary was not found or is not executable: ${resolved}`,
      );
    }
    return resolved;
  }

  const home = os.homedir();
  const exeNames = process.platform === 'win32' ? ['agy.exe', 'agy.cmd', 'agy'] : ['agy'];

  const wellKnownCandidates: string[] = [
    ...exeNames.map((n) => path.join(home, '.local', 'bin', n)),
    ...exeNames.map((n) => path.join(home, '.antigravity', 'bin', n)),
  ];

  if (process.platform === 'darwin') {
    wellKnownCandidates.push(
      '/Applications/Antigravity.app/Contents/Resources/app/bin/agy',
      path.join(home, 'Applications', 'Antigravity.app', 'Contents', 'Resources', 'app', 'bin', 'agy'),
      '/opt/homebrew/bin/agy',
      '/usr/local/bin/agy',
    );
  } else if (process.platform === 'linux') {
    wellKnownCandidates.push('/usr/local/bin/agy', '/usr/bin/agy');
  } else if (process.platform === 'win32') {
    const localAppData = env.LOCALAPPDATA ?? path.join(home, 'AppData', 'Local');
    wellKnownCandidates.push(
      path.join(localAppData, 'Programs', 'Antigravity', 'bin', 'agy.cmd'),
      path.join(localAppData, 'Programs', 'Antigravity', 'bin', 'agy.exe'),
    );
  }

  for (const candidate of wellKnownCandidates) {
    if (isExecutableFile(candidate)) return candidate;
  }

  const fromPath = searchPathForBinary(exeNames, env.PATH);
  if (fromPath) return fromPath;

  throw new Error(
    'dsh-agy-native: official `agy` CLI executable not found. Install Antigravity CLI (`~/.local/bin/agy`) and sign in with `agy login`, or set `agyPath` in plugin settings.',
  );
}

/**
 * Build a sanitized child environment for `agy acp --stdio` that guarantees
 * subscription OAuth credentials are used rather than ambient API keys.
 */
export function createSanitizedAgyEnv(baseEnv: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...baseEnv };
  for (const key of STRIPPED_AUTH_ENV_KEYS) {
    delete env[key];
  }
  return env;
}

/**
 * Return the Antigravity CLI state directory and verify whether it is writable.
 * A read-only `~/.gemini/antigravity-cli` causes `agy` token refresh during
 * `session/new` to stall indefinitely.
 */
export function checkAgyStateDirectory(homeDir = os.homedir()): { stateDir: string; writable: boolean } {
  const geminiDir = path.join(homeDir, '.gemini');
  const stateDir = path.join(geminiDir, 'antigravity-cli');
  try {
    if (fs.existsSync(stateDir)) {
      fs.accessSync(stateDir, fs.constants.R_OK | fs.constants.W_OK);
      return { stateDir, writable: true };
    }
    if (fs.existsSync(geminiDir)) {
      fs.accessSync(geminiDir, fs.constants.R_OK | fs.constants.W_OK);
      return { stateDir, writable: true };
    }
    fs.accessSync(homeDir, fs.constants.R_OK | fs.constants.W_OK);
    return { stateDir, writable: true };
  } catch {
    return { stateDir, writable: false };
  }
}

/**
 * Query `agy --version` quickly for diagnostics without touching network endpoints.
 */
export async function probeAgyVersion(agyPath: string, timeoutMs = 4000): Promise<string | undefined> {
  try {
    const { stdout, stderr } = await execFileAsync(agyPath, ['--version'], {
      timeout: timeoutMs,
      env: createSanitizedAgyEnv(),
    });
    const combined = `${stdout}\n${stderr}`.trim();
    return combined.split(/\r?\n/)[0]?.trim() || undefined;
  } catch {
    return undefined;
  }
}
