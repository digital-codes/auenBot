# Proposal

## Why

After a completion-style intent (e.g. `messdaten`) has answered, the
`intent` key stays in the sealed context forever. The router in
`botCtl.py` (`route_handler`) treats a context with `intent` set as
"continue this intent" and skips decoding entirely, so every later turn
in the session is executed against that stale intent. History
(`bot_history.db`, session `4711ebe6`, 17:05–17:07) shows the
consequences: after one O3 query, the inputs "was weißt du so", "wann
fährt der bus", "gibt es was zu essen" and "rheinauen" are all answered
with the messdaten option menu in an endless loop, and in the same
session the literal input "anreise" was swallowed by a stale
`tp_definition` intent. Intents that are known to work from a clean
context (anreise answered "wie komme ich zu euch" correctly in an
earlier session) are unreachable for the rest of a session once any
completion has run.

## What Changes

- **BREAKING (context contract)**: `intent` in the context is now a
  *pending-completion marker*: it is set only while a completion flow
  still waits for user input and is removed as soon as the flow
  completes (answer delivered). `last_intent`, `last_input` and
  `focus` remain the durable memory fields.
- A no-match against a pending completion (reply matches none of the
  offered options) no longer ends in an internal error string: the
  router falls back to re-decoding the input. If another intent is
  found, it takes over the turn and answers; only when re-decoding
  also fails does the no-match reply and miss counting apply.
- Unmatched-input error replies become user-appropriate text instead
  of leaking internal keys (`no_match_completion`) or placeholder
  strings ("Nichts gefunden... hier könnte ein Vorschlag kommen") —
  both observed verbatim as chat text in history rows 1214 and 1203.
- `botApiTest.py` gains regression scenarios: anreise/bus/verpflegung
  questions asked after a completed messdaten query reach their own
  intents.

## Capabilities

### New Capabilities

- `intent-lifecycle`: Owns how a turn is routed between a pending
  completion flow and a freshly decoded intent — when a pending intent
  consumes the input, when it is cleared after answering, and what
  happens when a pending flow cannot use the input.

### Modified Capabilities

- `conversation-recovery`: Clarify miss counting against the new
  no-match fallback — a turn rescued by re-decoding is an answer, not
  a miss; only turns that end without content count.
- `bot-api-testing`: Add regression coverage for post-completion turns
  reaching newly decoded intents.

## Impact

- `botCtl/botCtl.py` (`route_handler`): the `elif user_intent is not
  None ...` branch (lines ~562-564) must re-decode instead of blindly
  reusing the stale intent; option-resolution branch condition changes
  accordingly.
- `botCtl/botIntents.py`: `_finalize_completion` (and target-routing
  path) must pop `intent` from the returned context; `_make_error_response`
  callers for user-facing no-match replies need user-appropriate text.
- Context wire format: `intent` disappears from the sealed state after
  completed flows. `botQa.seal_state` consumers tolerant (opaque
  token); frontend renders `message`/`options` only — no impact.
- Tests: `botApiTest.py` extended; history DB schema unchanged.
