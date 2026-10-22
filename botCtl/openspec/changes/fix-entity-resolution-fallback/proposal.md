# Proposal

## Why

Auen-entity queries fail today in two ways: naming 6 of the 8 habitat
entities (`hartholzaue`, `stillgewässer`, `feuchtwiesen`, ...) yields
*"Nichts gefunden"* because the intent vectors of the `aue_*` family
never mention those names (only `Rheinaue` does), so the queries decode
into the `tp` cluster whose entity pool cannot contain them; and
`feuchtwiesen` additionally demonstrates that the stage-2 fuzzy fallback
invents confident wrong answers (`WRatio("feuchtwies", "efeu") = 77.1`
passes the 65 threshold — `luchs → rotfuchs` at 80.0 fires the same way).
An LLM arbiter on the failure path resolves both without touching the
precomputed intent vectors, which have no in-repo regeneration pipeline.

## What Changes

- **Fix 2 — stage-2 hardening**: the per-token fuzzy fallback in
  `BotIntent._match_input_to_items_structured` only accepts a candidate
  when the token stem and the candidate name share a stem prefix
  (first 3 characters). This kills the spurious high-WRatio class
  (`feuchtwies→efeu`, `luchs→rotfuchs`) while real typos survive
  (`feuchtwies→feuchtwiesen` at 90.9).
- **Fix 1 — LLM entity arbiter**: when the complete-handler flow cannot
  resolve the `entity` requirement (staged matcher returns nothing),
  a single `chat_json` call (temperature 0, same pattern as the existing
  `detectIntent` tie-breaker) classifies the user's term against the
  three domain candidate lists loaded from the existing
  `entity_tiere.json` / `entity_pflanzen.json` / `entity_aue.json`
  files. The returned name is validated strictly (exact, case-
  insensitive, member of the claimed domain's list); on success the
  entity requirement is filled and the pipeline continues —
  `find_entity` already resolves all 170 names including Auen rows. On
  `found: false`, invalid output, or LLM failure the existing
  *"Nichts gefunden"* error path is used unchanged.
- **Data stays as designed**: `entity_aue.json` remains the single
  owner of the Auen names (consulted as the arbiter's Auen candidate
  list); `entity_tp.json` is not merged and the precomputed intent
  vectors are not regenerated (no in-repo build script exists).
- `botApiTest.py` gains regression test cases: Auen resolution
  (`hartholzaue`, `stillgewässer`, `feuchtwiesen` expect real entity
  content), OOV no-match (`luchs` expects no species), and the existing
  10 samples plus the `frosch` scenario staying green.

## Capabilities

### New Capabilities

- `entity-arbiter`: The LLM-assisted entity recovery step that runs when
  staged entity matching fails — candidate lists, strict validation,
  pipeline continuation, and degradation semantics.

### Modified Capabilities

- `entity-resolution`: Stage-2 fuzzy fallback gains the shared-stem-
  prefix acceptance requirement; a rejected fallback no longer invents
  an entity.
- `bot-api-testing`: The harness gains entity-resolution regression
  samples for Auen resolution and OOV no-match behavior.

## Impact

- **Code**: `botIntents.py` (stage-2 prefix check in
  `_match_input_to_items_structured`; arbiter function + hook in
  `_handle_no_options`), `botApiTest.py` (new samples). No schema, API,
  or storage changes.
- **LLM usage**: one additional chat call only on entity-matching
  failure paths; `llm_used` propagates to the existing history column.
  LLM unavailability degrades to today's error behavior.
- **Data**: none — the three `entity_*.json` files and
  `intent_vectors.json` are unchanged.
- **Not affected**: state-token layer, sanitization, `botCtl.py` routing,
  `bot_history.db` schema, existing scenarios (`reh`, `frosch`,
  `entities`, security/restart suite).
