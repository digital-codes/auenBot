# Design

## Context

The turn pipeline (`botCtl.py`) decodes intents with `BotDecoder`
(bge-m3 embeddings, cosine kNN over 3687 vectors, confidence bands 0.25/0.75
with a chat-LLM arbiter) and completes `bio_feature` intents through the
staged German entity matcher in `botIntents.py`. Conversation state comes back
as a sealed token; `focus = {entity, type}` persists across turns. The Metis
spike (see proposal) validated 17 intents + 3 noul questions at 89.3% over 882
real example questions, stage-2 categories at conf 0.60–0.97 on clear cases.

Spike artifacts: `/tmp/opencode/metis_spike/` (`intents_simple.json`,
`category_map.json`, `spike.py`, `batch_test.py`, `batch_results.jsonl`).

## Goals / Non-Goals

**Goals:**
- One Metis call per non-bio turn, two per bio turn; no generated text.
- Keep the legacy decoder fully functional as fallback (flag-off or error).
- Keep the German entity matcher as the entity resolver and arbiter.
- German + scientific names only for criteria and state.

**Non-Goals:**
- Replacing entity resolution with Metis.
- Removing the embedding decoder, vectors or old intent files.
- Migrating the option menus (anreise/messdaten/expertise) to Metis answers.
- English/French locales; Metis path is German-only for now.

## Decisions

- **D1 — Flag-gated switchover, not replacement.** `private.py` gains
  `METIS_ROUTING` (bool); when unset, Metis is enabled iff `METIS_KEY` exists.
  With the flag off (or on any Metis error/timeout/non-200) the pipeline falls
  back to `BotDecoder` exactly as today. Rationale: the user tests both paths;
  the spike is not a regression test for the full bot.
- **D2 — Stage 1 decides intent + transitions jointly.** One request,
  one `choice` question (17 intents, German criteria descriptions from the
  spike) plus noul `followup`, `new_topic`, `unrelated`. State carries the
  user input, the focus entity name/type, and whether the previous question
  was answered. Routing order (API-test-validated):
  1. `smalltalk`/`beleidigung` p>0.5 → friendly path, focus cleared
  2. non-bio intent p<0.5 → intent options (legacy unsure path); a bio
     question stays in the bio flow even below 0.5 (bare entity names
     decode weakly but the entity search resolves them)
  3. other named intent → existing machinery for that intent
  4. `unrelated` ≥ 0.8 → fallback
  5. `followup` ≥ 0.7 and focus present → reuse focus entity, skip entity search
  6. otherwise → clear focus if `new_topic` ≥ 0.7, run entity search
  Focus anchoring guard: with a focus in the state Metis reads unrelated
  inputs as bio questions about the focused entity ("gibt es was zu essen"
  + focus Reh → bio_frage 0.83). A bio choice with followup below the
  threshold is therefore re-probed without the focus before routing.
  Alternative considered: followup as its own intent — rejected; noul
  decided jointly is cleaner and matches the existing gate semantics.
- **D3 — Stage 2 decides the feature category.** One `choice` question
  (verhalten/eigenschaften/beschreibung) over the resolved entity. Column
  mapping comes from `data/category_map.json`. Undecided = top-2 margin
  < 0.10 or confidence < 0.35 → merged overview of all filled columns plus
  2-3 random options drawn from the entity's actually filled features.
  Media rule: audio for verhalten, image otherwise (from `Links`).
- **D4 — Answer text stays mechanical for now.** The category answer returns
  the merged column data as today's `get_entity_features` does. An LLM that
  phrases the answer from category + question is a follow-up, not part of
  this change.
- **D5 — `intents_metis.json` maps new intents onto existing machinery.**
  Each coarse intent declares `handler`/`action`/`requires` reusing the
  existing `complete`/`bio_feature` flow; utterances are taken from the
  representative old intents (e.g. `anreise` keeps its utter). `bio_frage`
  requires only `entity`; the feature comes from stage 2.
- **D6 — New intents map to old ones by name, not by re-implementation.**
  Where a coarse intent groups old sub-intents (ausstellungen, kinder,
  fuehrungen, aktivitaeten), the coarse utterance of the parent intent is
  used; per-sub-intent utterances are not reachable via Metis routing.
- **D7 — Known accepted deviations.** Bot-persona favourites
  ("dein Lieblingstier") may decode to `bio_frage` (31/882 in the batch);
  they are answered via the LLM persona path, not canned bio data.
  `aue_hochwasser` questions route to `messdaten` (live data) — intended.
- **D8 — Timeouts.** Metis call timeout 60s (docs recommendation for the
  TypeSafe SDK); on 429/5xx one retry honouring `Retry-After`, then legacy
  fallback for that turn.

## Risks / Trade-offs

- Metis is an external EU service: outage → automatic legacy fallback, slight
  latency spike per turn before the fallback engages.
- The 17 criteria descriptions are load-bearing text; accuracy degrades if
  they drift without re-running the batch test.
- `data/intents_metis.json` and `data/intents_raw.json` coexist; intent ids
  reported in history differ per routing mode.
- Rate limit 600 requests / 5 min per account; a busy bot plus batch tests
  can exhaust it — batch scripts must throttle.
