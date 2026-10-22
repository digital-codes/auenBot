# Design

## Context

Entity resolution for intent requirements runs in
`BotIntent._match_input_to_items` (`botIntents.py`): the whole user
sentence is fuzzy-matched (`process.extractOne`, `fuzz.WRatio`) against
the items of the first missing requirement — for `tp_definition` that
is the 162 species names in `data/entity_tp.json`. For *"gibt es da
frösche"* this returns `Gelbes Windröschen` at 66.7 (threshold 65):
`WRatio` rewards the shared *rösche* substring, and no German
normalization exists that could relate *frösche* to the five
*frosch*-stem species in the list (there is no generic *Frosch*
entry). The resolved name is then written into `ctx["entity"]` and
re-verified by `botActions.find_entity`, which only re-confirms the
wrong pick.

The completion flow already has a working two-step selection
mechanism: `_handle_no_options` can put `ctx["options"]` (list of
`{"title": ...}` dicts) into the context and return a choose-prompt;
on the next turn `_handle_with_options` matches the user's reply
against those titles via `_match_input_to_options` and fills the
missing requirement. The group behavior chosen by the user
(plural → offer species options) maps directly onto this mechanism.

See proposal.md — Why, and specs/entity-resolution for the contract.

## Goals / Non-Goals

**Goals:**
- *Frösche*-type queries reach the *frosch*-stem species, never a
  lookalike name.
- German plural normalization as a shared, dependency-free helper.
- Group matches surface as selectable options through the existing
  options flow.
- Match discipline: stem containment outranks fuzzy similarity;
  low-confidence whole-sentence matches can no longer select entities.

**Non-Goals:**
- Full lemmatization or an NLP library (spaCy etc.) — a small rule
  set for German plural stems is sufficient for this data set.
- Changing `botActions.find_entity`/`_match_single_entity` behavior
  (bio_feature re-resolution); it receives specific names.
- Changes to data files, `botCtl.py`, or the API contract.
- LLM-based entity extraction (the pipeline's LLM use stays as is).

## Decisions

### D1: New shared normalizer module `botText.py`

Create `botText.py` with pure functions:
`german_stem(word) -> str` (lowercase, map ä→a/ö→o/ü→u, strip plural
suffixes in decreasing specificity: `-ern`, `-en`, `-er`, `-e`,
`-n`; keep a minimum stem length of 3) and
`normalize(text) -> [stems]` (tokenize on non-letters, stem each
token). No new dependency; easily unit-testable; importable by both
`botIntents.py` and later `botActions.py`.

Implementation refinement (found while verifying 2.1): the `-s`
suffix was dropped from the strip list. Repeated reduction turned
`Wiesen` into `wie`, colliding with the function word — and because
input and names stem symmetrically, `-s` differences are already
covered by the matcher's prefix/suffix containment, so `-s` only
added collisions. The stemmer reduces suffixes repeatedly until
stable, which makes it idempotent as required.

*Rationale:* umlaut reversion + suffix stripping turns *Frösche* →
`frosch` and leaves *Windröschen* → `windroschen` — the two names
that today collide become clearly distinguishable. *Alternative
considered:* rapidfuzz's built-in processors (no morphology),
Levenshtein on stems only (still substring-blind), or adding a
lemma list to the data files (data maintenance burden, doesn't fix
arbitrary plurals). Rule-based stemming wins on fit and cost for a
162-name closed vocabulary.

Known limitation: over-stripping (e.g. *Blumen* → `blum`,
*Buchen* → `buch`) only risks **missing** a match or lumping
stems that share a suffix-stripped form; it cannot invent a wrong
high-confidence match, and the group/options flow handles the
lumping case gracefully. Accepted and covered by the threshold
rules in D3.

### D1a: Evaluated — model-based reranking / LLM entity selection

Considered whether a reranking model or a chat-model rerank
instruction should replace or back the stem matcher. Findings from
a live probe of the configured endpoint (DeepInfra, 2026-10-22):

- **Reranking model: unavailable.** The endpoint exposes no rerank
  route (`/v1/rerank` → 404) and no rerank model in its catalog;
  only embedding (`BAAI/bge-m3`) and chat models. Adopting a
  reranker would require a provider change or self-hosting.
- **Chat model works semantically but is unfit as the primary
  matcher.** Measured with `mistralai/Mistral-Small-3.2-24B` on the
  real case: picking from the full 162-name list took ~2.7 s and
  was over-inclusive (returned the whole amphibian class — toads
  and newts — for *frösche*); picking from a small candidate list
  took ~0.8 s but **hallucinated an entity name that is not in the
  list** (`"Frosch"`); an unrelated query correctly returned `[]`.
- **Where a chat-model step could still pay off:** as a bounded
  semantic rescue *after* stem and fuzzy matching both fail (e.g.
  paraphrases or non-German input), following the existing
  intent-disambiguation pattern in `botDecoder.detectIntent`
  (candidate list → `chat_json` with a strict instruction → validate
  output against the item list → graceful fallback on error).

Decision: keep the rule-based stemmer as the primary mechanism
(deterministic, offline, ~0 ms, and it scopes the group to exactly
the *frosch*-stem species — something the full-list LLM call did
not manage). An LLM fallback is deliberately **not** part of this
change; it is a candidate follow-up if stem+fuzzy misses are
observed in practice. If added later it MUST validate the model's
output against the item list (hallucinated names observed) and MUST
keep the existing failure fallback.

### D2: Restructure `_match_input_to_items` matching order

Replace the single whole-sentence `extractOne` with a staged
matcher inside `botIntents.py` (entity resolution path only; option
titles keep using the existing simple path):

1. Build stems for every item once per call (items are ≤ 162
   strings; no caching needed).
2. **Stage 1 — exact stem containment:** for each input token stem,
   collect items whose full-name stem equals it, or whose
   multi-word name stem contains it as a word-stem (e.g. `frosch`
   matches the stem of *kleiner Wasserfrosch* via its second word).
3. **Stage 2 — fuzzy fallback:** only if stage 1 is empty, run the
   existing `WRatio` match, but per input token against item names
   (not whole sentence), and accept only scores ≥
   `self.matchThreshold`.

Return value becomes a structured result: either a single item, a
group (list of items), or none — instead of one string. The public
behavior for single matches is unchanged (callers still receive a
matched name).

Implementation refinements (found while verifying against the
tiere/pflanzen/auen tables):

- Loose prefix/suffix containment (rank 2) accepts token stems of
  3+ characters, guarded by a function-word stem list
  (`botText.FUNCTION_WORD_STEMS`) — short words like *wie*, *das*,
  *gibt* would otherwise spuriously match *Wiesen-...* stems.
  This lets genuine short terms (*aue*, *auen*) match the Auen
  table.
- Containment runs against both word stems and the raw lowercase
  words: *Rheinaue* stems to `rheinau` (trailing *e* stripped), so
  *aue* only matches on the raw form.
- A query can legitimately span two groups (*"wiesen in der aue"*);
  the matcher then returns the union as one option list — the user
  disambiguates, per the group contract.

*Rationale:* containment on stems is what actually encodes
"frösche ⇒ frosch family"; whole-sentence WRatio is what produced
the bug. *Alternative:* fix only the threshold (e.g. 75) —
rejected: it trades one wrong match for dropped valid matches and
still has no path from *frösche* to *frosch*.

### D3: Group match → existing options flow

In `_handle_no_options`, when the matcher returns a group of > 1
items for the `entity` requirement: build `ctx["options"]` as
`[{"title": name} for name in group]` (the shape
`_match_input_to_options` already reads), set `ctx["intent"]`, and
return a choose-prompt output (reuse the `select_options` message).
The entity requirement stays unfilled; on the next turn
`_handle_with_options` resolves the user's pick exactly as it does
for configured option files. A single match fills the requirement
directly, as today.

Group members are ordered by descending match quality (exact
word-stem equality first) so the prompt lists the most likely
species first.

*Rationale:* zero new state shapes for the app — `ctx["options"]`
with `title` entries is the established contract (`botCtl.py` and
the front-end already handle it). *Alternative:* new context key
`entity_group` with custom handling — rejected as redundant
plumbing on both sides.

### D4: Regression check via the HTTP harness

Extend the manual verification path: run `botApiTest.py` with a
*"gibt es da frösche"* request and assert the response offers the
options (context contains the five *frosch* species, `entity` not
set), then send *teichfrosch* as the follow-up and assert the
flow completes with `Teichfrosch`. The existing Reh scenario stays
the guard against regressions in single-entity resolution.

## Risks / Trade-offs

- [Rule-based stemming mis-stems unusual species names] → Stemming
  is symmetric (input and names), minimum stem length 3, and stage-1
  equality on full names still wins; worst case is a fallback to
  fuzzy matching, i.e. today's behavior minus the whole-sentence
  failure mode.
- [Group prompt adds a turn for plural queries] → This is the
  explicitly chosen UX; singular/species queries are unaffected.
- [`_match_input_to_items` is shared by other requirement types
  (e.g. feature items)] → The staged matcher keys on the
  requirement being `entity`-like list content; for non-entity
  lists stage 1 still applies harmlessly (stem containment on short
  feature words), and stage 2 preserves the old semantics. Verified
  by the existing scenario runs.
- [Umlaut reversion collides distinct names (ä→a)] → Only affects
  stem *containment* candidates, which are then listed as a group
  for the user to disambiguate rather than silently resolved.

## Migration Plan

Deploy is a service restart; no schema, data, or API changes.
Rollback is reverting the commit. No backfill: nothing persisted
depends on the old match values beyond conversation history rows.

## Open Questions

None.
