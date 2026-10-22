# Proposal

## Why

The `bot_error.log` run for *"gibt es da frösche"* shows the entity
resolution picking **Gelbes Windröschen** — a plant — for a question
about frogs. The intent (`tp_definition`) and feature (`definition`)
are detected correctly; the defect is the entity match in
`BotIntent._match_input_to_items` (`botIntents.py`): it fuzzy-matches
the **whole user sentence** against the 162 entity names from
`data/entity_tp.json` with `fuzz.WRatio`, and *frösche* is lexically
closer to *Windröschen* than to any item name, so the wrong match
clears the threshold (66.7 ≥ 65) and poisons the whole response.

Two underlying defects make this failure mode systematic, not a one-
off:

1. **No German morphology.** Entity names are singular
   (*Teichfrosch, Springfrosch, kleiner Wasserfrosch, Moorfrosch,
   Laubfrosch* exist in the data; there is no generic *Frosch*
   entry). A plural like *frösche* can never match any of them —
   neither exactly nor by stem — while fuzzy whole-sentence matching
   happily returns unrelated names that happen to share substrings
   (*rösche*).
2. **No group resolution.** When a user query names a group that
   covers several species (plural *Frösche*, or a genus-like term),
   the pipeline has no way to narrow it down; the only mechanism it
   knows is a single best match, which for a group query is
   arbitrary.

The user chose the target behavior: a plural/group query SHALL
offer the matching species as selectable options (the existing
options flow) rather than silently picking one.

## What Changes

- Add a shared, rule-based German token normalizer (no new
  dependency): lowercase, umlaut-stem mapping (ä→a, ö→o, ü→u) and
  German plural-suffix stripping (-e, -en, -n, -er, -s, -ern) so
  *Frösche* and *Frosch* reduce to the same stem.
- Restructure the entity matching in
  `BotIntent._match_input_to_items` so entity queries are matched at
  the token level against normalizer-reduced entity names — exact
  stem containment first, fuzzy scoring only as fallback — instead
  of whole-sentence `extractOne`. Threshold discipline: a
  whole-sentence fuzzy score alone must no longer select an entity.
- Add group detection: when one input stem matches **multiple**
  entity items (e.g. all *…frosch* species), the matcher returns the
  group. In the completion flow (`_handle_no_options`), a group
  match SHALL present the group members as options (`ctx["options"]`
  with `title` entries, reusing the existing options mechanism and
  `_handle_with_options` selection on the next turn) instead of
  filling the `entity` requirement arbitrarily. A single match
  fills the requirement as today.
- Keep `botActions.find_entity`/`_match_single_entity`
  (bio_feature re-resolution) unchanged in contract; it receives
  specific entity names as before. The shared normalizer is placed
  so it can be reused there, but changing its behavior is not part
  of this change.

## Capabilities

### New Capabilities

- `entity-resolution`: Capability that owns mapping German user
  input to entity items for intent requirements: token-level
  matching with German plural normalization, exact-stem matching
  before fuzzy fallback, and group handling that offers the
  matching species as options when one query stem spans several
  entities.

### Modified Capabilities

(None. `bio-feature-lookup`'s single-best-entity contract is
untouched — by the time `bio_feature` runs, the entity requirement
is filled with one specific entity, either matched directly or
selected from a group's options.)

## Impact

- `botIntents.py`: `_match_input_to_items` (token/stem matching),
  `_handle_no_options` (group → options presentation), and the
  context/requirement flow around it. `_handle_with_options` is
  reused unchanged.
- New shared normalizer (small module, e.g. `botText.py`, or a
  clearly separated helper section in `botActions.py`) — no new
  pip dependency.
- `data/*.json` item files: unchanged; the normalizer works on the
  existing 162 names.
- `botCtl.py`: unchanged — group options ride the existing
  `ctx["options"]` flow already handled by `route_handler` and the
  app.
- Observable regression check: replay *"gibt es da frösche"* via
  `botApiTest.py` — the response must offer the five frog species,
  and must not mention *Windröschen*.
