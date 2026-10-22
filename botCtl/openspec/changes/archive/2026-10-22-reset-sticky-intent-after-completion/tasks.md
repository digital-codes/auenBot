# Tasks

## 1. Pending-intent marker lifecycle (botIntents.py)

- [x] 1.1 In `_finalize_completion`, pop `intent` from the returned context (keep `last_intent`, `last_input`, `focus`, and requirement removal as-is). Verify: execute a completion-style intent to completion via `botIntents.py` debug main and confirm the returned context has no `intent` key but retains `last_intent`/`focus`.
- [x] 1.2 Pop `intent` on the target-routing path in `_finalize_completion` as well (it rewrites the context to `{intent: target, last_input}` before executing the routed intent — the routed intent's own completion result must again be marker-free). Verify: complete an intent whose requirements include `target` and confirm the final context has no `intent`.
- [x] 1.3 Confirm `_handle_no_options`/`_handle_with_options` still set `ctx["intent"]` while options are pending (no change, add a comment stating the marker semantics: set while waiting, popped on completion). Verify: present-options path returns context with `intent` and `options` both set.

## 2. Routing precedence and no-match fallback (botCtl.py)

- [x] 2.1 Restructure `route_handler` decision chain to: decline/reset → option resolution (gated on options being present, not on empty intent) → pending intent execution → fresh decode (`decoder.detectIntent`) → fallback intent. Verify: `python3 -c "import botCtl"` imports; manual POST of a pending-options session with an option title completes the flow as before.
- [x] 2.2 When the pending intent execution returns `no_match`, re-run `decoder.detectIntent(user_input)`; if a different intent is found, execute it for this turn (its result replaces the failed one; clear pending options). Verify: API test below (3.2) passes.
- [x] 2.3 Keep miss counting semantics: a rescued turn resets the counter; a turn that also fails re-decoding counts as a miss exactly as today (including the breakout at the configured limit). Verify: three consecutive gibberish inputs on a pending-options session trigger the breakout message and an empty next context.

## 3. User-facing error text (botIntents.py)

- [x] 3.1 Replace the user-visible no-match error texts with one localized, user-appropriate sentence for the unmatched-input paths that currently leak `no_match_completion` or the "Nichts gefunden... hier könnte ein Vorschlag kommen" placeholder; keep returning the internal key via the result (`error` field) for history/logging. Verify: trigger a no-match turn and confirm the TX message text is the human-readable sentence and history `status`/`error` still identify the cause.
- [x] 3.2 Sweep remaining `_make_error_response` call sites for other keys reachable from user input and confirm none can send a raw key as chat text (internal-only failures like `options_file_not_found` may keep a descriptive error, but must not read like debug output). Verify: grep call sites; note the decision per key in the PR description.

## 4. Regression tests (botApiTest.py)

- [x] 4.1 Add scenario: complete a messdaten query (menu → O3/NO2 pick), then send "wann fährt der bus"; assert the reply addresses the bus/anreise topic and the messdaten menu is not re-presented. Verify: run the test file.
- [x] 4.2 Add scenario: complete a tp/tiere/pflanzen question, then send "anreise"; assert the anreise flow starts and the sealed state contains no `intent`. Verify: run the test file.
- [x] 4.3 Add scenario: with the messdaten menu pending, send "gibt es was zu essen"; assert the verpflegung reply and that no internal error key appears in the text. Verify: run the test file.
- [x] 4.4 Add scenario: gibberish while options are pending three times triggers the existing breakout (no behavioral change, guards D2/D3). Verify: run the test file.

## 5. Full verification

- [x] 5.1 Run the whole `botApiTest.py` suite against a locally started service and confirm all scenarios (old and new) pass. Verify: test run output.
- [x] 5.2 Replay the failing history sequence manually (messdaten → O3 → "was weißt du so" → "wann fährt der bus" → "gibt es was zu essen" → "rheinauen") and inspect `bot_history.db`: no repeated option menu, no internal error keys, each question answered by its own intent. Verify: sqlite query over the new rows.
