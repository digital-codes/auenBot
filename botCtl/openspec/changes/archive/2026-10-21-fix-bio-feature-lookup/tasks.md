# Tasks

## 1. Reproduce the bug and confirm root cause

- [x] 1.1 Start `botCtl.py -d` and run `python3 botApiTest.py
  --scenario reh`. Capture the printed `message.text` and confirm
  it contains `Köcherfliegen` / `Regenbremse` substrings alongside
  `Reh` content. Verify by `grep -E "Köcherfliegen|Regenbremse"`
  over the captured output and confirming at least one match.

- [x] 1.2 Add a one-off Python REPL probe that calls
  `BotAction.find_entity("Reh")` directly against the live data
  file and prints the returned entity list plus each entry's
  `Name`. Verify that the list contains *more than one* entity
  (currently `Reh` and `Regenbremse`), confirming the multi-entity
  return is the root cause of the mixing.

## 2. Fix `BotAction.find_entity`

- [x] 2.1 Replace the four `idxs = [targets.index(match[0]) for
  match in matches]` blocks in `BotAction.find_entity` (the
  `entity_type is None`, `Tier`, `Pflanze`, and `Auen` branches)
  with `idxs = [match[2] for match in matches]`, since
  `process.extract` returns `(choice, score, index)`. Verify by
  re-running the REPL probe from task 1.2 and confirming `Reh`
  resolves to the same `self.data[147]` entry before and after.
  Implementation note: the per-branch logic was factored into a
  shared `_match_single_entity` helper that applies the same
  fix in one place.

- [x] 2.2 Make `BotAction.find_entity` return only the single
  best-scoring entity (after the existing threshold + Typ filter)
  in all four branches. Verify by re-running the REPL probe from
  task 1.2 and confirming `find_entity("Reh")` returns a list with
  exactly one entity whose `Name == "Reh"`.

- [x] 2.3 Replace the noisy `print("Matches found:", matches)` /
  `print("Matches above threshold:", matches)` /
  `print("Entities found:", ents)` blocks with three concise
  single-line debug prints per branch: candidates (name + score),
  post-filter candidate list, and final picked entity. Verify by
  re-running the REPL probe from task 1.2 with `DEBUG=True` and
  confirming exactly one "picked entity" line is printed for the
  `Reh` query, naming `Reh`.

## 3. Tighten `BotAction.get_entity_features`

- [x] 3.1 Add a one-line debug print inside the per-key loop in
  `BotAction.get_entity_features` showing the key, the resolved
  entity, and the chosen value (or `None`). Verify by re-running
  the REPL probe from task 1.2 with `DEBUG=True` and confirming
  the loop emits one line per feature key, all naming `Reh`.

- [x] 3.2 Confirm the existing "no values for any key" fallback
  still triggers correctly: with `DEBUG=True`, manually call
  `get_entity_features("Reh", "nonexistent_key")` against the live
  data and confirm the response contains the German
  *"leider keine Informationen vorhanden"* message rather than
  empty lists.

## 4. Extend `botApiTest.py` with a PASS/FAIL line

- [x] 4.1 In `scenario_reh` (in `botApiTest.py`), join the returned
  `message.text` into one string and print a single `PASS` /
  `FAIL: <reason>` line before the existing `=== end ===` marker.
  PASS requires the joined text to contain at least one of
  `Ricke`, `Kitz`, `Rehbock`, `Capreolus`, `Geweih` AND to contain
  neither `Köcherfliegen` nor `Regenbremse`. Verify by running
  `python3 botApiTest.py --scenario reh` against a `botCtl` that
  has been restarted to pick up the `botActions.py` changes, and
  confirming the line reads `PASS`.

- [x] 4.2 Sanity-check the FAIL branch by temporarily reverting
  the `BotAction.find_entity` fix on a scratch copy, restarting
  `botCtl`, and re-running `python3 botApiTest.py --scenario reh`.
  Confirm the script prints `FAIL: response mentions Köcherfliegen`
  (or the actual offending substring). Verify by reading the
  script's FAIL output and confirming it names the offending
  species. Restore the fix immediately afterwards.

## 5. End-to-end verification

- [x] 5.1 Run the full smoke test against a clean `botCtl.py -d`
  restart: `python3 botApiTest.py` exits 0 AND prints `PASS`.
  Verify by capturing `echo $?` after the run (expect `0`) and
  `grep -E "^PASS$"` over the output (expect one match).

- [x] 5.2 Read the final `botCtl.log` from this run and confirm
  the new debug output shows exactly one entity picked for the
  `Reh` query, with no `Regenbremse` or `Köcherfliegen` line in
  the picked-entity trace. Verify by `grep -c "Regenbremse"`
  over the picked-entity lines and confirming the count is `0`.

## Workflow follow-up

- Archive this change after the *Reh* scenario prints `PASS` and
  the new debug output looks right; the test script
  `botApiTest.py` becomes the standing regression check for the
  bio_feature contract going forward.
