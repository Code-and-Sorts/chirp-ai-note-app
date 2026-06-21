# Security Policy

## Supported versions

Chirp is pre-1.0 (`0.0.1a0`). Security fixes land on `main` and ship in the
next release. Only the latest published version is supported.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub Security Advisories
("Report a vulnerability" on the repository's **Security** tab) rather than a
public issue. We aim to acknowledge reports within a few days.

## Known advisories / risk acceptances

Chirp runs entirely on-device: it embeds ChromaDB as a local `PersistentClient`
and uses PyTorch only for local voice-activity detection via `silero-vad`. It
never starts a network server and never executes untrusted model code. The
following dependency advisories have **no fixed release available** and are not
reachable in Chirp's usage, so they are accepted rather than patched. The
corresponding Dependabot alerts may be dismissed as "vulnerable code is not
actually used."

| Dependency | Advisory | Why Chirp is not affected |
|---|---|---|
| chromadb | GHSA-f4j7-r4q5-qw2c (CVE-2026-45829) — pre-authentication code injection via `trust_remote_code` on the ChromaDB **HTTP server** endpoint `/api/v2/tenants/{tenant}/databases/{db}/collections` | Chirp uses ChromaDB only as an embedded, in-process `PersistentClient` (`notes_chat/index.py`). It does not run `chromadb` as a server, does not expose any HTTP API, and never sets `trust_remote_code`. The vulnerable endpoint is unreachable. No patched release exists (latest 1.5.9 is still flagged). |
| torch | GHSA-rrmf-rvhw-rf47 (CVE-2025-3000) — local memory corruption reachable through `torch.jit.script` | Pulled in transitively by `silero-vad` for on-device VAD. Chirp never calls `torch.jit.script`, and only feeds locally recorded audio to the bundled VAD model — there is no untrusted-input path to the affected function. No patched release exists. |

Fixable advisories are remediated by upgrading; for example
`pydantic-settings` (GHSA-4xgf-cpjx-pc3j) is pinned to `>=2.14.2` in
`pyproject.toml`.
