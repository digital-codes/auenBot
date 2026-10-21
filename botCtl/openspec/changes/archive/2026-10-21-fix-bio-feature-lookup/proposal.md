# Proposal

## Why

The archived change `add-bot-api-test-script` (2026-10-21) added
`botApiTest.py`, which makes it trivial to drive a running `botCtl`
over HTTP. Running the `reh` scenario against the live service
immediately surfaces a bug the prior log captures had been hinting
at: for the input *"was macht das Reh"* the bot correctly identifies
intent `tp_definition` and resolves the entity to `Reh`, but the
returned `message.text` mixes descriptions of unrelated entities
(*Köcherfliegen*, *Regenbremse*) with the actual *Reh* feature
fields. The user-facing answer therefore contains claims about
species the user never asked about.

The mixing is the most visible symptom. Reading the call path
(`botCtl.py` → `BotIntent.execute` → `_finalize_completion` →
`BotAction.handle_bio_feature` → `BotAction.get_entity_features` →
`BotAction.find_entity`) reveals two contributing defects in
`botActions.py`:

1. `BotAction.find_entity` returns **every** entity above the
   configured match threshold (currently 65, set in `botCtl.py`),
   not just the best match. For *"Reh"* the run yields both
   `reh` (100%) and `regenbremse` (72%), which after the
   `Typ=='Tier'` filter still leaves two Tier entities.
2. `BotAction.get_entity_features` then iterates **all** of those
   items and concatenates their feature fields, so Regenbremse's
   description text is emitted alongside Reh's. The
   `targets.index(match[0])` lookup in `find_entity` (with the
   correct `match[2]` form sitting commented out on the next line)
   is a related stale anti-pattern but, for *Reh*, is not the root
   cause — it would only matter when entity names collide.

Because the test harness already exists and the symptom is
observable, we can now close the loop: investigate, fix the root
cause, optionally enrich debug output to make the fix self-evident,
and verify against the *Reh* scenario.

## What Changes

- Fix the root cause in `botActions.py` so the `bio_feature` action
  returns features for the **single best-matching** entity of the
  requested `Typ`, not a concatenation across all above-threshold
  matches.
- Clean up the stale `targets.index(match[0])` index-recovery lines
  in `BotAction.find_entity` and use the index `process.extract`
  already provides (`match[2]`) — the correct form is sitting
  commented out on the next line of all four match branches.
- Tighten the debug output in `BotAction.find_entity` and
  `BotAction.get_entity_features` so the chosen entity, the
  filtered candidates, and the per-key chosen values are printed
  on one line each — this makes the post-fix run obviously correct
  and the pre-fix run obviously wrong, without needing extra
  logging glue in `botCtl.py`.
- Re-run the `reh` scenario from `botApiTest.py` and capture the
  response: the returned `message.text` MUST mention *Reh* content
  and MUST NOT mention *Köcherfliegen*, *Regenbremse*, or any
  other non-*Reh* species.
- **BREAKING**: callers of `BotAction.find_entity` that depend on
  the multi-entity return shape must move to the new single-entity
  contract. No in-repo callers exist today
  (`botCtl.py` only uses `get_entity_features`); flagged here so a
  future change that re-uses `find_entity` directly does not get a
  surprise.

## Capabilities

### New Capabilities

- `bio-feature-lookup`: Capability that owns the contract for the
  `bio_feature` action's input resolution and feature retrieval.
  Covers how a user input is mapped to a single entity, how the
  feature key is mapped to the data fields, and how the response is
  structured (text / image / audio) so callers can rely on it
  returning features for the resolved entity only.

### Modified Capabilities

- `bot-api-testing`: A small, additive update to the
  "Reh scenario surfaces the bio-feature output" requirement so
  the script prints a clear PASS/FAIL line against the post-fix
  expectations. Behaviour of the test harness itself does not
  change; only what the scenario asserts about the bot's output.

## Impact

- `botActions.py`: fix `find_entity`/`get_entity_features`,
  strengthen debug output. Same public surface
  (`execute`, `handle_*`, `get_entity_features`), so `botCtl.py`
  and other callers do not change.
- `botApiTest.py`: extend the `reh` scenario to print a PASS/FAIL
  line against the expected post-fix behaviour, but keep the
  full-text dump for human inspection.
- No data files changed; no vector files changed; no new
  dependencies.
- The fix is observable end-to-end through the existing
  `botApiTest.py` script, which becomes the regression check.
