# Tasks

## 1. Metis client

- [x] 1.1 Create `botMetis.py`: config from `private.py` (`METIS_KEY`,
      `METIS_URL`, `METIS_MDL`, optional `METIS_ROUTING`), `enabled()` check,
      `_ask(state, questions)` with 60s timeout and one retry on 429/5xx
      honouring `Retry-After`
- [x] 1.2 `detect(query, history)`: stage-1 request (17-option intent choice +
      followup/new_topic/unrelated noul), returns parsed answers dict
- [x] 1.3 `category(query, entity_name)`: stage-2 request, returns category,
      probabilities, tie flag (margin < 0.10 or confidence < 0.35)

## 2. Simplified intent and data files

- [x] 2.1 Create `data/intents_metis.json`: 17 intents with German criteria
      descriptions from the spike, utterances reused from representative old
      intents, `bio_frage` with `handler=complete`, `requires=entity`,
      `action=bio_feature`
- [x] 2.2 Create `data/category_map.json`: verhalten/eigenschaften/beschreibung
      column groups from the spike

## 3. Pipeline integration

- [x] 3.1 `botCtl.py`: load Metis config; when enabled, replace the
      `_decode_new_intent` embedding path with `botMetis.detect` and map the
      coarse intent to the matching intent record from
      `data/intents_metis.json`; keep legacy decoder as fallback on flag-off
      and on any Metis error
- [x] 3.2 Apply the routing order (smalltalk/beleidigung → unrelated → other
      intents → followup focus reuse → new_topic focus clear + entity search)
      with thresholds p>0.5, 0.8, 0.7
- [x] 3.3 `botIntents.py`: on followup with focus, inject the focused entity
      before requirement matching; on new_topic, clear the focus
- [x] 3.4 `botActions.py`: extend `get_entity_features` with the category
      column groups; build merged-category answers with media attachment
      (audio for verhalten, else image); on undecided, merged overview plus
      2-3 random options from the entity's filled features

## 4. Verification

- [x] 4.1 `metis_test.py`: smoke test running ~10 German queries (bio clear,
      follow-up, topic switch, smalltalk, unrelated, NAZKA) through the new
      routing against the live endpoint — 20/20 PASS
- [x] 4.2 Legacy path check: with `METIS_ROUTING=False`, existing behaviour
      unchanged (decoder used, bio feature flow unchanged) — boot check:
      no bio_frage record appended, decoder with 3687 vectors loaded
- [ ] 4.3 Run `botApiTest.py` happy path against the flag-off server
      (deferred: needs the live service; owner will test)
