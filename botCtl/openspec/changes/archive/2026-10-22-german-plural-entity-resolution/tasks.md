# Tasks

## 1. German normalizer module

- [x] 1.1 Create `botText.py` with `german_stem(word)` (lowercase, ä→a/ö→o/ü→u mapping, plural suffix stripping `-ern`, `-en`, `-er`, `-e`, `-n`, `-s` in that order, minimum stem length 3) and `normalize(text)` (tokenize on non-letters, return stem list). Verify with a REPL check: `german_stem("Frösche") == "frosch"`, `german_stem("Windröschen") == "windroschen"`, `german_stem("Moorfrosch") == "moorfrosch"`, `normalize("gibt es da frösche")` contains `"frosch"`, and stemming is idempotent (`german_stem(german_stem(w)) == german_stem(w)` for a sample list).

## 2. Staged entity matching in `botIntents.py`

- [x] 2.1 Extend `_match_input_to_items` with a staged matcher per design D2: build item stems, stage 1 exact stem containment (full name stem equality, then word-stem containment for multi-word names), stage 2 per-token fuzzy fallback with `fuzz.WRatio` at `self.matchThreshold`; return a structured result (single / group / none). Verify with a direct call against `data/entity_tp.json`: `"gibt es da frösche"` → group of the five *frosch* species and **no** *Windröschen*; `"moorfrosch"` → single `Moorfrosch`; `"windröschen"` → `Gelbes Windröschen`/`Busch-Windröschen` group or best single per containment rules.
- [x] 2.2 Wire the structured result through the completion flow in `_handle_no_options`: single match fills the requirement as today; group of > 1 sets `ctx["options"] = [{"title": name} ...]` (ordered by match quality), sets `ctx["intent"]`, returns the `select_options` prompt, and leaves the requirement unfilled; no match keeps the current error response. Verify by simulating `_handle_no_options` for the frog query and asserting the returned context contains exactly the five species options and no `entity` key.
- [x] 2.3 Confirm the follow-up turn works end-to-end through the unchanged `_handle_with_options`: user reply *teichfrosch* against the offered options fills `entity` with `Teichfrosch` and finalizes. Verify by simulating `_handle_with_options` with `ctx["options"]` from 2.2 and input `teichfrosch`.

## 3. Regression and live verification

- [x] 3.1 Run the non-entity matching paths (option titles, feature items) to confirm no regression: `_match_input_to_options` with a title list and the existing Reh scenario via `botApiTest.py`. Verification: both behave as before (Reh scenario PASS, option selection still resolves).
- [x] 3.2 Start the Flask service and replay the failing conversation via `botApiTest.py`: send *"gibt es da frösche"*, assert the response asks for a choice and the context options are exactly the five *frosch* species with no `entity` in context; then send *"teichfrosch"* and assert the flow completes with `Teichfrosch`. Verification: the service log shows the group detection and no `Windröschen` resolution for the run.

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Verify the archived result.
