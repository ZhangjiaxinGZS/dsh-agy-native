#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-web}"
DSH_BIN="${HOME}/.local/bin/dsh"
PROFILE_DIR="${HOME}/.dsh/profiles/${PROFILE}"
PATCH_FILE="${PROFILE_DIR}/cordis.patch.yml"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "==> [1/4] Building & packing dsh-agy-native in ${REPO_DIR}..."
cd "${REPO_DIR}"
npm run build
TGZ_NAME="$(npm pack --silent | tail -n 1)"
TGZ_PATH="${REPO_DIR}/${TGZ_NAME}"
echo "    Packed: ${TGZ_PATH}"

echo "==> [2/4] Backing up existing profile config..."
mkdir -p "${PROFILE_DIR}"
if [[ -f "${PATCH_FILE}" ]]; then
  cp "${PATCH_FILE}" "${PATCH_FILE}.bak-${STAMP}"
  echo "    Backed up ${PATCH_FILE} -> ${PATCH_FILE}.bak-${STAMP}"
fi
if [[ -f "${DSH_BIN}" ]]; then
  cp "${DSH_BIN}" "${DSH_BIN}.bak-${STAMP}"
  echo "    Backed up ${DSH_BIN} -> ${DSH_BIN}.bak-${STAMP}"
fi

echo "==> [3/4] Installing ${TGZ_NAME} into DSH profile '${PROFILE}'..."
"${DSH_BIN}" plugin --profile "${PROFILE}" remove -w dsh-gemini-oauth-bridge >/dev/null 2>&1 || true
"${DSH_BIN}" plugin --profile "${PROFILE}" add -w "${TGZ_PATH}"

echo "==> [4/4] Writing clean cordis.patch.yml (removing dsh-carbon-test hacks)..."
cat > "${PATCH_FILE}" <<'YAML'
- id: agent-default-model
  config:
    provider: agy-acp
    model: gemini-4-carbon
- id: ui-settings-general
  name: "@deepseek-ai/dsh-client-ui-settings-general"
  config:
    welcomeNoticeVersion: 2026-09-28.1
YAML

# Clean DSH_AGY_LOCAL_KEY injection from ~/.local/bin/dsh if present
if [[ -f "${DSH_BIN}" ]] && grep -q "dsh-carbon-test/loopback.key" "${DSH_BIN}"; then
  EXEC_LINE="$(grep '^exec ' "${DSH_BIN}" | head -n 1)"
  if [[ -n "${EXEC_LINE}" ]]; then
    printf '#!/bin/sh\n%s\n' "${EXEC_LINE}" > "${DSH_BIN}"
    chmod +x "${DSH_BIN}"
    echo "    Cleaned loopback.key wrapper from ${DSH_BIN}"
  fi
fi

echo ""
echo "✅ Switched DSH profile '${PROFILE}' to dsh-agy-native!"
echo "   Restart your DSH Web server when ready (e.g. restart 'dsh web') and refresh http://127.0.0.1:3080."
