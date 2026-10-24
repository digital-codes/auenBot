# Proposal

## Why

The intent decoder (3687 embedding vectors over 93 intents) plus the chat-LLM
arbiter and the follow-up gate need several model round-trips per turn, and the
93-intent surface is hard to maintain. A spike (batch over 882 real example
questions, 4 iterations) showed the GreenPT decision model `metis` routes a
turn with 89.3% accuracy in a single forward pass when the intent surface is
simplified to 17 coarse intents plus 3 yes/no transition questions, and picks
the answer feature category in a second pass. This replaces the embedding
decoder and the follow-up LLM gate while keeping the proven German entity
search as arbiter and data source.

## What Changes

- Add `botMetis.py`: a Metis client with two stages —
  stage 1 decides intent (17-option choice) jointly with three noul questions
  (followup, new_topic, unrelated); stage 2 decides the feature category
  (behaviour / properties / description) with an undecided fallback.
- Add simplified intent/data files (`data/intents_metis.json`,
  `data/category_map.json`): smalltalk, beleidigung, bio_frage, nazka_info,
  anreise, ausstellungen, kinder, aktivitaeten, essen, hunde, fuehrungen,
  barrierefreiheit, toiletten, wetter, messdaten, expertise, fallback.
- Wire Metis routing into the `/api` turn pipeline behind a config flag
  (`METIS_ROUTING` in `private.py`, default on when a `METIS_KEY` is present);
  the existing embedding decoder stays as fallback when the flag is off or
  Metis fails.
- Follow-up handling: when stage 1 says followup and a focus entity exists,
  the focused entity is reused and the German entity search is skipped;
  new_topic clears the focus and runs the entity search.
- Feature answers for `bio_frage` come from stage 2: merged data of the chosen
  category column group, media attachment (image for properties/description,
  audio for behaviour) and, on undecided (close margin or low confidence),
  a compact overview plus 2-3 random feature options.
- Old intent files, vectors and the embedding decoder remain untouched and
  selectable; nothing is removed.

## Capabilities

### New Capabilities

- `metis-routing`: Metis-based turn routing — stage-1 intent choice with
  joint followup/new_topic/unrelated decisions, stage-2 feature-category
  decision with undecided fallback, thresholds, and fallback to the legacy
  decoder on error or when disabled.

### Modified Capabilities

- `intent-lifecycle`: with Metis routing enabled, follow-up/new-topic
  transition decisions come from stage-1 noul answers instead of the LLM
  follow-up gate; the pending-completion capture rules stay unchanged.
- `bio-feature-lookup`: with Metis routing enabled, the feature for a
  bio question is the stage-2 category (behaviour/properties/description)
  answered with merged column data plus media, instead of one
  intent-suffixed feature column.

## Impact

- New: `botMetis.py`, `data/intents_metis.json`, `data/category_map.json`,
  `metis_test.py` (smoke test).
- Modified: `botCtl.py` (decode path + config), `botIntents.py` (category
  features, followup focus injection), `botActions.py` (merged category
  answer building).
- Runtime dependency: GreenPT API (`POST /v1/systemone`, model `metis`,
  EU endpoint) with `METIS_KEY`; one call per non-bio turn, two per bio turn.
- The spike scripts live outside the repo (`/tmp/opencode/metis_spike/`).
