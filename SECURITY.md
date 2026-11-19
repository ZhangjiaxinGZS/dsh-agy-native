# Security & Account Compliance Policy

`dsh-agy-native` is engineered around a strict **Zero-TOS-Violation & Local-Only Trust Boundary** model.

## 1. Account & Terms-of-Service Safety Invariants

1. **100% Official Binary Egress**:
   - This package contains **zero** OAuth client IDs, **zero** client secrets, and **zero** direct HTTP calls to `cloudcode-pa.googleapis.com` or any Google internal API.
   - All upstream authentication, token refresh, request signing, telemetry envelopes, and streaming calls are performed exclusively by the user's locally installed, Google-signed `agy` executable (`agy acp --stdio`).
2. **100% Untouched Official `systemInstruction` & Circuit Breaker**:
   - Removing Antigravity's default system prompt components (`excludeDefaultComponents: true`) or injecting third-party agent persona headers (`"You are an AI agent powered by..."`) into `request.systemInstruction` triggers Cloud Code Assist edge WAF `429 RESOURCE_EXHAUSTED` / `403 TOS_VIOLATION` flags ([#1597](https://github.com/google-antigravity/antigravity-cli/issues/1597)).
   - `dsh-agy-native` **never** sets `excludeDefaultComponents: true` **and never mutates `_meta.antigravity.systemPrompt`**. Workspace operational guidelines are neutralized and delivered cleanly inside the initial user turn (`contents`).
   - If `agy` ever encounters an upstream `429 RESOURCE_EXHAUSTED` / `403 PERMISSION_DENIED` response, `dsh-agy-native` trips an **instant attempt-1 circuit breaker** (`session/cancel` + non-retryable `QUOTA` error) so neither `agy` nor DSH ever hammers Google with multi-attempt retry storms.
3. **Ambient API Key Isolation**:
   - `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `AGY_ADC_AUTH`, and `DSH_AGY_LOCAL_KEY` are stripped from the `agy` child environment before spawning so ambient environment variables cannot cause accidental pay-per-token billing or credential cross-contamination.

## 2. Local Loopback MCP Bridge Hardening

1. **Loopback-Only Binding**:
   - The Streamable-HTTP MCP bridge binds strictly to `127.0.0.1:0` (never `0.0.0.0`) and is never reachable from external networks.
2. **Per-Session 192-bit Cryptographic Capability Token**:
   - Every ACP session receives a unique `crypto.randomBytes(24).toString('hex')` bearer header (`x-dsh-mcp-token`). Any request without a valid active session token is rejected with `401 Unauthorized`.
3. **Strict Tool Allowlist & Permission Gate**:
   - `agy`'s built-in tools are disabled (`enabledTools: []`).
   - Incoming ACP `session/request_permission` requests are automatically approved **only** when the tool name matches an active DSH tool registered for that exact session; any unrecognized tool request is automatically rejected (`reject_once`).
4. **Argument Sanitization**:
   - Auxiliary fields injected by `agy` (such as undeclared `toolSummary` properties) are stripped before passing arguments back to DeepSeek Harness, preserving strict JSON Schema (`additionalProperties: false`) guarantees.

## Reporting Vulnerabilities

Please report security issues via [GitHub Security Advisories](https://github.com/weijiafu14/dsh-agy-native/security/advisories/new).
