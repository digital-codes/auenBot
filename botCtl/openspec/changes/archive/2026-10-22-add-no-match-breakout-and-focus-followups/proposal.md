# Proposal

## Why

A user whose input keeps failing entity resolution is trapped: the
context keeps the decoded intent set, so every following input re-enters
the same completion flow, and repeated misses produce an endless loop of
*"Nichts gefunden"* with no exit except guessing a valid entity name or
a hardcoded decline word. Conversely, a user who just got a good answer
("was macht das reh") cannot ask a natural follow-up ("und wie groß ist
sie?") — the follow-up decodes to a feature intent, entity resolution
fails, and the flow answers *"Nichts gefunden"* even though the entity
was resolved one turn earlier. The `qAndA` prototype and the fork's
`QaEngine` both demonstrate the two missing mechanisms; both can be
built on the existing stateless context round-trip without touching the
sealed-token wire contract.

## What Changes

- **Configurable no-match breakout**: a `no_match_count` in the
  conversation context increments on every no-match outcome (matcher +
  arbiter failed, or an offered option was never matched) and resets on
  any successful match. When the count reaches a configurable limit
  (default 3, settable via a `setMaxNoMatch` setter like the existing
  `setThreshold`), the context is cleared entirely (intent, slots,
  focus — a full conversation restart under the same session id) and
  the reply invites the user to ask again. The counter is implemented
  centrally in `route_handler` so both no-match sites are covered.
- **Focus-based follow-ups**: after every successful completion, the
  resolved entity and type are stored as a `focus` snapshot in the
  context. When a later turn decodes to a slot-filling intent
  (tp_/tiere_/pflanzen_ family), the entity requirement fails staged
  matching *and* the arbiter, and a focus exists, the entity is filled
  from the focus (direct fill, no confirmation turn) and the pipeline
  proceeds — the feature comes from the new intent, so "und wie groß
  ist sie?" after a Reh answer returns the Reh's size.
- **`last_input`/`last_intent` become live**: populated on completion
  (the schema fields and the reading code already exist; the writes are
  commented out today) so the round-tripped context carries the trace
  the follow-up and history rows imply.
- **Feature-fallback options**: when a resolved entity has no data for
  the requested feature, the reply informs the user and offers up to 3
  of that entity's data-carrying features as selectable options (via
  the existing options flow); picking one answers with that feature's
  content. Covers e.g. the size follow-up on Reh (no `Größe` field) and
  the `tp_generell` dead end.
- `botApiTest.py` gains scenarios: breakout after the configured number
  of misses (context verified empty afterward, via the decrypted token),
  follow-up flow ("was macht das reh" → "und wie groß ist sie?" expects
  the informed fallback with feature options, picking one yields real
  content), and the existing 15 samples plus all other scenarios
  staying green.

## Capabilities

### New Capabilities

- `conversation-recovery`: The breakout and follow-up mechanisms —
  miss counting with configurable threshold, context reset semantics,
  focus snapshot lifecycle, and focus-based entity recovery.

### Modified Capabilities

- `bot-api-testing`: The harness gains breakout and follow-up
  scenarios as regression samples.

## Impact

- **Code**: `botCtl.py` (`route_handler` counter/reset hook,
  `setMaxNoMatch` wiring), `botIntents.py` (`_finalize_completion`
  focus snapshot, `_handle_no_options` focus fill, setter), 
  `botApiTest.py` (new scenarios). No API contract, schema, or storage
  changes — the new context keys travel inside the sealed `state`
  token and appear in the plaintext history as before.
- **Interaction**: changes observable bot behavior (breakout message
  replaces repeated error loops; follow-ups get answered instead of
  "Nichts gefunden"). The decline-word list in `route_handler` remains
  and still clears intent independently of the breakout.
- **Not affected**: `botQa.py` sealing, sanitization, entity matcher
  and arbiter logic, `bot_history.db` schema, all existing scenarios.
