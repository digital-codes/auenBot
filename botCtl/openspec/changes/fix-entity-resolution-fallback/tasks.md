# Tasks

## 1. Stage-2 hardening

- [x] 1.1 In `botIntents.py` `_match_input_to_items_structured`, accept a stage-2 fuzzy candidate only when the token stem and the candidate name's stem share their first 3 characters (umlaut-mapped on both sides); verify by direct call: `feuchtwies` vs `entity_tp.json` → no match (Efeu rejected), `feuchtwies` vs `entity_aue.json` → Feuchtwiesen accepted, and a differently-prefixed typo (`morfrosch`) is rejected by stage 2 (the arbiter recovers it in task 2)
- [x] 1.2 Verify no regression on the calibrated matcher cases: `was macht das reh`, `gibt es da frösche`, `windröschen`, `knabenkraut`, `moorfrosch` produce identical structured results against the tp pool as before the change

## 2. LLM entity arbiter

- [x] 2.1 Add the arbiter to `botIntents.py`: load the three candidate lists from `data/entity_tiere.json` / `entity_pflanzen.json` / `entity_aue.json`, one `chat_json` call (temperature 0, schema hint, JSON output `{domain, name, found}`) classifying the user input and picking the referenced entity name; verify by direct call with `hartholzaue` (→ Auen/Hartholzaue), `luchs` (→ found false), and an unavailable endpoint (→ exception surfaced)
- [x] 2.2 Add strict validation: accept only a name that matches exactly one entry of the claimed domain's file (case-insensitive, trimmed, domain consistent); verify hallucinated names, cross-domain claims, and unknown domains are all rejected by unit-style direct calls
- [x] 2.3 Hook the arbiter into `_handle_no_options`: on staged-matcher miss, run the arbiter, fill the `entity` requirement from the validated answer, propagate `llm_used=true` into the turn's history rows, and fall through to the existing no-match error response on any arbiter failure; verify end-to-end via the running service: `hartholzaue` resolves with real Erkennungsmerkmale content and the history row carries `llm_used = true`, `luchs` yields the normal no-match behavior

## 3. Harness regression samples

- [x] 3.1 Extend `botApiTest.py` `ENTITY_SAMPLES` with the Auen and OOV cases: `hartholzaue`, `stillgewässer`, `feuchtwiesen`, `magerrasen` (expect species-content replies, not the no-match error) and `luchs` (expect no species resolution); verify each new sample PASSes against the running service in attach mode
- [x] 3.2 Verify the full suite stays green: `python botApiTest.py --scenario all` exits 0 with all scenarios PASS (reh, frosch, entities incl. new samples, tampered, foreign, db_crosscheck, restart_resume, restart_no_secret)

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Open design question deferred as follow-up: habitat-specific feature mapping for Auen queries (intent modeling change).
