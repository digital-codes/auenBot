# Tasks

## 1. Port the token layer

- [x] 1.1 Copy `botQa.py` from the fork (`CodeforKarlsruhe/auenBot`, `main`) into `botCtl/`, drop the `QaEngine`/FSM/`HistoryEvent` half, and keep only the seal/open section; verify `python -c "import botQa"` succeeds
- [x] 1.2 Add the explicit `secret=` parameter to `seal_state`/`open_state` (defaulting to the module-level `STATE_TOKEN_SECRET`-derived secret) and keep `seal_state_tuple`/`open_state_tuple` for internal reuse; verify a round-trip `open_state(seal_state({...}, "s1"), "s1")` returns the original dict and a wrong secret raises
- [x] 1.3 Add `cryptography` to the project environment and verify `python -c "from cryptography.hazmat.primitives.ciphers.aead import AESGCM"` succeeds

## 2. API contract and backend wiring

- [x] 2.1 Update `botSchema.json`: add `state` (string), remove `context`; verify `jsonschema` validation of a sample new-session request (no `state`) and a continuation request (with `state`) against the updated schema
- [x] 2.2 In `botCtl.py` `check_input()`: open the incoming `state` token with the request's session id; on failure return/abort with HTTP 400 (tampered, malformed, foreign session) and never fall back to plaintext context; verify with a manual curl round: valid token → 200, flipped character → 400
- [x] 2.3 In `botCtl.py` response paths (`route_handler` intent-execution response and the option-prompt early return): seal the outgoing context into a fresh `state` token and return it instead of `context`; verify the response JSON contains `state` and no `context` key
- [x] 2.4 Confirm plaintext storage is unchanged: run one full turn and verify `bot_history.db` contains RX and TX rows whose `context` column is readable plaintext JSON with intent/entity values
- [x] 2.5 Verify the startup secret behavior: with `STATE_TOKEN_SECRET` unset the service logs the random-secret warning and still works; with it set across a restart a pre-restart token is accepted

## 3. Input sanitization

- [x] 3.1 Port the fork's text validation into `check_input()`: reject HTML tags, the character class `` ` { } | $ ; & < > ``, and C0 control characters except tab/newline/CR, all with HTTP 400 before any pipeline processing; verify each rule with a curl request returning 400
- [x] 3.2 Verify umlauts and normal German text pass unchanged: send `"was macht das reh in der aue?"` and confirm status 200 and correct intent resolution

## 4. Harness: spawn, secret, and token handling

- [x] 4.1 Add service lifecycle to `botApiTest.py`: `--start`/`--no-start` (spawn default), `--secret` (default: generate random hex), child `Popen` with `STATE_TOKEN_SECRET` in env, health polling until `GET /api` returns ok, teardown in `finally`; verify a full run starts the service, runs one scenario, and exits without leaving the child alive
- [x] 4.2 Update `ScenarioState`: replace `context` with the `state` token on the wire, add a `context()` accessor that decrypts the current response token via `botQa.open_state` with the harness secret, and make `send()` seal nothing (first request) / echo the token (continuations); verify the existing `reh` scenario still prints PASS using the decrypted context
- [x] 4.3 Keep attach mode working: with `--no-start --secret <hex>` against an externally started service (started with the same secret in its env), verify the harness runs scenarios without spawning a child

## 5. Harness: security and restart scenarios

- [x] 5.1 Add the tampered-token scenario: flip one character in the token part, expect HTTP 400, print PASS/FAIL; verify it fails loudly (FAIL) if the service ever accepts a modified token
- [x] 5.2 Add the foreign-session scenario: run two sessions in parallel, send session A's token in session B's request, expect HTTP 400; verify PASS on the built-in fix
- [x] 5.3 Add the same-secret restart scenario: complete a turn, restart the child with the same fixed secret, continue with the pre-restart token, expect the conversation to resume; verify PASS
- [x] 5.4 Add the missing-secret restart scenario: restart the child without `STATE_TOKEN_SECRET`, send a pre-restart token, expect HTTP 400; verify PASS
- [x] 5.5 Add the DB cross-check: after a turn, read the latest RX/TX rows for the session via `sqlite3` and assert stored plaintext context equals the decrypted wire context and the row count increased by two; verify PASS

## 6. Integration verification

- [x] 6.1 Run every scenario end-to-end against a single spawned instance: `python botApiTest.py --scenario reh` plus all new scenarios in one invocation, all reporting PASS; verify exit code 0
- [x] 6.2 Verify the full contract from the outside with a scripted curl flow (new session → token → continuation → tamper → 400) and record the observed behavior against the `state-token-encryption` spec scenarios

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Notify the frontend owner of the breaking `context` → `state` contract change before deployment (lockstep release).
- Decide at apply time where `STATE_TOKEN_SECRET` lives in deployment (env file vs. `private.py` field).
