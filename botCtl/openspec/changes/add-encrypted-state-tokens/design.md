# Design

## Context

The bot is already stateless by design: the frontend echoes `session`,
`sequence`, and a plaintext `context` object back on every request
(`check_input()` in `botCtl.py` trusts it verbatim); `bot_history.db` is a
write-only audit log, never read to reconstruct conversation state. The
`CodeforKarlsruhe/auenBot` fork contains a working prototype of the sealed
state token mechanism (`botCtl/botQa.py`): AES-256-GCM tokens keyed by
`HMAC-SHA256(STATE_TOKEN_SECRET, session_id)` with the session id bound as
AAD, plus a per-event sealing demo. The fork's `botCtl.py` additionally
carries request text sanitization that this branch lacks. The fork itself
is stale relative to `odd26` (old history code, no `botLlm`/`botText`/
`botApiTest`, no OpenSpec), so the port is done file-by-file, not by
merging. See proposal.md for the full motivation.

## Goals / Non-Goals

**Goals:**

- Wire-only confidentiality and integrity of conversation state: the
  client carries an opaque token it can neither read nor forge; the server
  works on decrypted plaintext.
- Keep the backend architecture exactly as it is (stateless, SQLite audit
  log stays plaintext).
- Make the security behavior testable end-to-end through `botApiTest.py`,
  including the frontend's side of the contract (seal what you send, open
  what you receive).
- Input text sanitization at the API boundary.

**Non-Goals:**

- `QaEngine` slot-filling FSM and per-event history tokens (candidate
  follow-up change reusing the same token plumbing).
- Data-at-rest encryption of `bot_history.db`.
- Replay protection via sequence binding (tokens do not expire in this
  change; see Risks).
- Porting other fork conveniences (`USE_DB` Postgres vector mode, decline-
  intent handling, `-t` CLI loop, `qAndA/` FastAPI experiment).
- Frontend implementation work (contract change is documented; the
  frontend repo is outside this project).

## Decisions

### D1: File port, not git merge

`botQa.py` is copied from the fork's `main` and adapted locally; the fork's
`botCtl.py` is not merged. The fork diverged badly and carries unrelated
changes; a merge would drag in regression-prone history code this branch
already fixed. `botQa.py` is self-contained (only `cryptography`), so the
port is clean.

### D2: Sealed context blob (one token per conversation turn), not per-event tokens

The whole context is sealed into a single `state` string per response.
Per-event sealing (`seal_state_tuple` in the fork demo) is only needed by
`QaEngine`'s event-log replay, which is out of scope. The blob keeps the
payload constant-size and the frontend change minimal. `QaEngine` can later
switch to a list of tokens without changing the key derivation or open/
seal semantics.

### D3: `secret=` parameter on seal/open; module secret as fallback

The fork reads `STATE_TOKEN_SECRET` into a module global at import time.
`botQa.py` is refactored during the port so `seal_state`/`open_state`
accept an explicit `secret=` argument, defaulting to the module-level
derived secret. Rationale: the harness (and any multi-process deployment)
must not depend on setting an env var before import; the import-order trap
is otherwise silent (every decrypt fails with an opaque InvalidTag).

### D4: Server decrypts, stores plaintext, re-seals

`check_input()` opens the incoming token into `ctx` and the pipeline
continues unchanged; `store_history` keeps writing plaintext rows
(existing `bot-history-storage` behavior unchanged); each response path
seals the outgoing `ctx` into a fresh token (fresh nonce per turn).
Alternative considered: store the sealed token per row for supportability
— rejected for now, plaintext rows already serve audit/debug and the
contract change is smaller.

### D5: `state` replaces `context` on the wire (breaking, no dual-read window)

`botSchema.json` gains a required-for-continuation `state` (string) and
loses `context`. A transition window accepting both was considered and
rejected: this is a two-party internal API with a test harness that
simulates the frontend, so lockstep deployment is realistic, and keeping
the plaintext path would leave the forgery hole open. New sessions (no
token yet) send only `session: ""` / omit `state`, as today.

### D6: Harness spawns the service and holds the secret

`botApiTest.py` grows: `--secret` (fixed hex secret; default generates
one), spawn of `python botCtl.py -d` with `STATE_TOKEN_SECRET` in the
child env (poll `GET /api` until ready; models load at import, so allow
a generous timeout), teardown in `finally`. Attach mode (`--no-start`)
still works: secret must be supplied or taken from the environment.
The harness imports `botQa` and uses `secret=` (D3) for white-box
seal/open. Existing `ScenarioState.context` assertions keep working by
decrypting the returned token on demand.

### D7: Sanitization rules ported as-is

The fork's regex rules move into `check_input()` unchanged: reject HTML
tags (`<...>`), the character class `` ` { } | $ ; & < > ``, and C0
control characters except tab/newline/CR — all with `400 invalid
request`. Non-goal note: this blocks `{`/`}`, which is fine for a Q&A bot
whose answers never echo braces. Umlauts and unicode letters pass.

## Risks / Trade-offs

- [Plaintext audit DB still holds full conversation state] → Accepted
  explicitly: the threat model is wire tampering and client-side
  inspection, not disk access; documented in the proposal.
- [Captured tokens replay forever (no expiry, sequence not bound)] →
  Mitigation candidate for a follow-up: include `sequence` in the AAD and
  reject non-increasing sequences. Not silently solvable in this change
  because the response re-seal happens after the pipeline mutates state;
  kept out of scope consciously.
- [`cryptography` is a new runtime dependency] → Widely packaged; the
  port needs no other new imports.
- [Breaking contract needs a simultaneous frontend drop] → The harness
  simulates the frontend and the change is small (echo a string); the
  frontend repo owner must be notified before apply starts.
- [Tokens die on restart when `STATE_TOKEN_SECRET` is unset] → Fork
  behavior preserved deliberately, with the startup warning; deployment
  docs must set the env var. The harness tests both modes (D6).
- [Model load at import makes spawn slow (~model download/loading time)]
  → Harness polls health with a long timeout and reuses one spawned
  instance across all scenarios.

## Migration Plan

1. Port `botQa.py` (D1, D3), add `cryptography` to the environment.
2. Update `botSchema.json`; wire open/seal into `check_input()` and the
   response paths; port sanitization (D7).
3. Extend `botApiTest.py` (D6) and run all scenarios green, including the
   DB cross-check and the restart pair.
4. Deploy with `STATE_TOKEN_SECRET` set; notify frontend of the breaking
   contract change (lockstep release, D5).
5. Rollback: revert the branch — the DB is unaffected, and the frontend
   reverts to plaintext context echo.

## Open Questions

- Where should `STATE_TOKEN_SECRET` live in deployment (systemd env file,
  `.env`, or a `private.py` field alongside the LLM credentials)? Does not
  affect the design; decide at apply time.
