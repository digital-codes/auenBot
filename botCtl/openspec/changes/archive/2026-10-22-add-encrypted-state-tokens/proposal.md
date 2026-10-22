# Proposal

## Why

The conversation context travels as plaintext and is trusted blindly by the
backend: the frontend receives `context` on every response and the backend
accepts whatever context arrives in the next request. Any client can inspect
the conversation state and forge fields such as `intent`, `entity`, or
`feature` to steer the pipeline. Porting the sealed-state-token mechanism from
the `CodeforKarlsruhe/auenBot` fork (`botCtl/botQa.py`) makes the wire format
opaque and tamper-evident while keeping the backend stateless — the state the
client carries is ciphertext it cannot read or modify, and the server keeps
storing decrypted plaintext history in `bot_history.db` as today.

The fork also carries input text sanitization (rejection of HTML tags,
injection-prone characters, and control characters) that `odd26` lacks;
adding it costs little and closes an obvious input-hardening gap while the
API contract is being touched anyway.

## What Changes

- Port `botQa.py` from the fork as `botQa.py` on this branch — the
  seal/open half only: AES-256-GCM tokens (`v1.<base64url(nonce||ct)>`),
  key derived via HMAC-SHA256 from `STATE_TOKEN_SECRET` + session id,
  session id bound as AAD. The `QaEngine` slot-filling FSM is **out of
  scope** (follow-up change).
- **BREAKING**: The API request/response contract changes. Requests carry
  an opaque `state` token instead of a `context` object; responses return
  a freshly sealed `state` token instead of the plaintext `context`.
  `botSchema.json` is updated accordingly.
- `check_input()` in `botCtl.py` opens (decrypts) the incoming state token;
  tampered, wrong-session, or malformed tokens are rejected with `400`.
  Response paths re-seal the outgoing context.
- SQLite history storage is unchanged in behavior: rows are stored in
  decrypted plaintext exactly as today (the encryption is wire-only).
- Port input sanitization from the fork's `botCtl.py` into `check_input()`:
  reject HTML tags, backticks/braces/pipe/dollar/semicolon/ampersand/
  angle-bracket characters, and C0 control characters (except tab/newline/
  CR) in the message text with `400`.
- `botQa.py` seal/open functions accept an explicit `secret=` parameter
  (defaulting to the module-level `STATE_TOKEN_SECRET`-derived secret) so
  callers and tests are not bound to import-time environment reads.
- `botApiTest.py` is extended into a test orchestrator: it can start
  `botCtl.py` itself with a generated or `--secret`-provided
  `STATE_TOKEN_SECRET`, seals requests / opens responses (white-box
  decryption) so existing context assertions keep working, and gains new
  scenarios: tampered token → 400, foreign-session token → 400, restart
  with fixed secret → session resumes, restart without secret → old token
  rejected, and a `bot_history.db` cross-check asserting stored plaintext
  matches the decrypted wire context.

## Capabilities

### New Capabilities

- `state-token-encryption`: The sealed state token layer — token format,
  key derivation from server secret + session id, session binding via AAD,
  open/seal behavior, and rejection semantics for tampered, foreign, and
  stale tokens (including after restart without a configured secret).
- `request-sanitization`: Validation rules applied to incoming message
  text (HTML tags, injection-prone characters, control characters) and
  their rejection behavior at the API boundary.

### Modified Capabilities

- `bot-api-testing`: The HTTP test harness gains the ability to start the
  service itself with a shared `STATE_TOKEN_SECRET`, to seal requests and
  open responses with that secret, and to drive new security and restart
  scenarios (tampered/foreign tokens, secret-dependent session resume,
  plaintext-DB cross-check).

## Impact

- **Code**: `botCtl.py` (`check_input`, response paths in `route_handler`),
  new `botQa.py` (ported), `botSchema.json`, `botApiTest.py`.
- **API contract**: **Breaking** for the existing frontend — `context`
  object replaced by opaque `state` token in both directions. The frontend
  change is small (echo a string instead of an object) but must land in
  lockstep with the backend.
- **Dependencies**: `cryptography` (AESGCM) added to the service runtime.
- **Environment**: `STATE_TOKEN_SECRET` env var (optional; random secret
  with a warning when unset, tokens then invalid across restarts).
- **Database**: none — `bot_history.db` schema and plaintext storage are
  unchanged (`bot-history-storage` spec unaffected).
- **Not affected**: `botIntents.py`, `botActions.py`, `botLlm.py`,
  `botText.py`, `botDecoder.py` — the pipeline below the API boundary.
