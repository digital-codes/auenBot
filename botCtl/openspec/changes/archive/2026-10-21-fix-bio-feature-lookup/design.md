# Design

## Context

The bot's `bio_feature` action lives in `botActions.py` and is
invoked by `BotIntent.__completeHandler` once all required context
slots are filled. The relevant methods today are:

- `BotAction.handle_bio_feature` — dispatches on the `bio_feature`
  handler name, reads `entity` / `feature` from the context, and
  delegates to `get_entity_features`.
- `BotAction.find_entity` — fuzzy-matches the user query against
  the entity list, currently returns every match above threshold,
  filters by `Typ`, and is the suspected site of the bug.
- `BotAction.get_entity_features` — looks up the entity by name
  (calling `find_entity` a second time), iterates **every** returned
  entity, and concatenates each entity's feature fields into one
  response.

`botCtl.py` only consumes the public surface of `BotAction`
(`execute`, `handle_*`, `get_entity_features`), so an internal fix
is safe. The archived change `add-bot-api-test-script` already
provides the regression harness: `botApiTest.py` plus the
`scenario_reh` function. See `proposal.md` for motivation and
`specs/bio-feature-lookup/spec.md` for the requirements this
design must satisfy.

## Goals / Non-Goals

**Goals:**

- Make `bio_feature` return features for exactly one entity per
  request, with the entity being the best-scoring match.
- Keep `botCtl.py` and the `botApiTest.py` public surface
  unchanged (no API or schema breakage).
- Strengthen debug output in `BotAction.find_entity` /
  `BotAction.get_entity_features` so the single-entity decision is
  obvious from a single bot run.
- Verify the *Reh* case end-to-end through `botApiTest.py`.

**Non-Goals:**

- Changing the entity data files (`rawData/tiere_pflanzen_auen.json`
  and friends) or the vector file.
- Re-tuning the fuzzy-match threshold — the current value is fine
  once the *consumer* of the matches only takes the best.
- Reworking the intent layer (`BotIntent`/`BotDecoder`) — the bug
  sits below it.
- Adding a new scenario or rewriting `botApiTest.py` beyond the
  small PASS/FAIL extension already scoped.

## Decisions

### Decision 1: Pick the single best match in `find_entity`

After the threshold filter, `BotAction.find_entity` sorts the
remaining matches by score descending and returns only the first
one. The `Typ` filter still runs after the threshold filter so
non-matching-type candidates are dropped first; the score sort
then picks the highest-scoring entity of the requested type.

- **Why**: this is the smallest change that satisfies
  *bio_feature resolves to a single best-matching entity*. The
  current `Typ` filter is preserved, so a plant match does not
  shadow an animal match for an animal query.
- **Alternative considered**: return the top N matches (e.g. top
  3) so callers can choose. Rejected because no current caller
  uses more than one and `get_entity_features` is the only
  consumer; the spec's "single best-matching entity" requirement
  also rules it out.
- **Alternative considered**: a separate "list candidates" method
  on `BotAction`. Deferred — out of scope for this bug fix.

### Decision 2: Replace `targets.index(match[0])` with `match[2]`

Each match returned by `process.extract` is
`(choice, score, index)` where `index` is the position in the
original `targets` list. Replace the
`idxs = [targets.index(match[0]) for match in matches]` lines (all
four match branches in `find_entity`) with
`idxs = [match[2] for match in matches]`. The correct form is
already commented out on the next line of the `Tier=None` branch.

- **Why**: the current code finds the *first* occurrence of
  `match[0]` in `targets`. For unique entity names this happens to
  be correct, but it is a footgun: any duplicate name returns the
  wrong record, and the `try/except: idxs = []` swallows the
  exception if the name is missing entirely, masking the bug.
- **Alternative considered**: keep the lookup but add a uniqueness
  check. Rejected — using the index `process.extract` already
  provides is simpler and matches the documented contract.
- **Alternative considered**: delete the `try/except`. Rejected
  for now — once `match[2]` is in place the lookup cannot fail, so
  the bare `idxs = [match[2] for match in matches]` can stand on
  its own, but we keep a narrow `except IndexError` for safety
  rather than removing exception handling entirely.

### Decision 3: Simplify `get_entity_features`

Today `get_entity_features` calls `self.find_entity(name)` and
loops over every returned entity. After Decision 1, `find_entity`
returns one entity, so the loop is naturally a single iteration. We
keep the loop structure (no control-flow rewrite) but rely on
`find_entity` to return one entity, and we extend the per-key loop
with a debug line that prints the resolved entity and the value
chosen for that key. The "no values for any key" branch stays.

- **Why**: minimal change, reuses the existing structure, and the
  per-key debug line makes the *single-entity* decision visible in
  the log without restructuring.
- **Alternative considered**: rewrite `get_entity_features` to
  take the already-resolved entity by reference. Rejected — the
  call chain from `botCtl.py` currently re-resolves by name and
  re-resolving works correctly after Decision 1; changing the
  signature would ripple through `botCtl.py` and `botActions.py`
  for no behaviour gain.

### Decision 4: Strengthen debug output in `find_entity`

Replace the existing `print("Matches found:", matches)` /
`print("Matches above threshold:", matches)` /
`print("Entities found:", ents)` lines with three concise,
single-line prints per match branch:

1. `Candidates for '<term>' (type=<t>): <[(name, score), ...]>`
2. `After Typ filter and best-match pick: <name> (score=<n>)`
3. (already added in Decision 3) one line per feature key showing
   the chosen value for that single entity.

- **Why**: makes the bug-fix verifiable from one log run. The
  pre-fix log would show two candidate entities; the post-fix log
  shows exactly one.
- **Alternative considered**: use `logging` instead of `print`.
  Rejected — the rest of the project uses `print` for debug
  output and switching one class to `logging` is out of scope.

### Decision 5: Extend `scenario_reh` with a PASS/FAIL line

Add an assertion at the end of `scenario_reh` (in `botApiTest.py`)
that joins `message.text` into one string and checks it for the
expected post-fix behaviour: contains `Reh`-related markers (at
least one of `Ricke`, `Kitz`, `Rehbock`, `Capreolus`, `Geweih`)
and does NOT contain `Köcherfliegen` or `Regenbremse`. Print a
single `PASS` / `FAIL: <reason>` line before the existing
`=== end ===` marker.

- **Why**: makes the script a regression check that is cheap to
  run by hand or in CI. Matches the new
  `bot-api-testing` MODIFIED scenario added by this change.
- **Alternative considered**: split into a separate
  `scenario_reh_post_fix` and gate via a CLI flag. Rejected —
  over-engineered for one extra line of output.
- **Alternative considered**: assert on the data shape (single
  entity, single set of keys). Rejected — too brittle against
  future data additions; checking the actual user-visible output
  is the contract we care about.

## Risks / Trade-offs

- [Risk] Reducing `find_entity` to a single match changes the
  shape of its return value; any caller using `len(ents) > 1` or
  indexing past `[0]` would break. → **Mitigation**: searched the
  repo — `botCtl.py` only consumes `get_entity_features`, and the
  in-file `__main__` block iterates `result[0]` only. No external
  callers.
- [Risk] The threshold (currently 65) is intentionally low to allow
  fuzzy matching for short German animal/plant names. If a single
  name has multiple plausible matches (e.g. *Frosch* vs
  *Wasserfrosch* vs *Laubfrosch*), the user gets only the
  highest-scoring one. → **Mitigation**: that is the user-visible
  contract this spec encodes; ambiguity should be resolved by the
  intent (`tp_definition` with `entity` slot) or by an options
  prompt, not by returning a mix. Out of scope here.
- [Risk] The commented-out correct index line (`match[2]`) on
  line 366 of `botActions.py` suggests the author already knew
  about this. There may be a reason it was reverted. →
  **Mitigation**: the comments above it explain nothing; the
  current behaviour is clearly wrong for any non-unique entity
  name. If the apply phase finds a contrary signal it should
  pause and surface it.
- [Trade-off] `find_entity` loses its multi-entity return shape
  (BREAKING per the proposal). No in-repo callers, so the trade
  is acceptable and documented in the proposal's *What Changes*.

## Migration Plan

- No deployment steps: only Python source files in the project
  tree change. `botCtl.py` and the data/vector files are
  untouched.
- Rollback: `git checkout` the changed files
  (`botActions.py`, `botApiTest.py`) and restart `botCtl.py`. The
  pre-fix behaviour returns immediately; the new
  `botApiTest.py` PASS line will FAIL on rollback, which is the
  intended signal.
- Verification: start `botCtl.py -d`, then run
  `python3 botApiTest.py --scenario reh`. Pre-fix: the script's
  FAIL line names `Köcherfliegen`/`Regenbremse`. Post-fix: the
  PASS line.

## Open Questions

- None. The exact buggy lines are visible in `botActions.py`
  (`targets.index(match[0])` and the multi-entity return) and the
  spec already encodes the post-fix contract.
