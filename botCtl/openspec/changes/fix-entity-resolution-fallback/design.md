# Design

## Context

Both defects were reproduced and traced (read-only probes against the
live matcher and vector data):

- The `aue_*` intents use handler `default` (static text, no entity
  resolution) — `entity_aue.json` is currently read by no code path.
  The complete-handler flow only ever loads
  `data/entity_{intent_base}.json`: `tp` → `entity_tp.json`
  (162 Tier+Pflanze names, 0 Auen names), `tiere` → 98, `pflanzen` → 64.
- Vector routing cannot rescue Auen queries: only "Rheinaue" appears in
  `aue_*` training vectors; the other 7 habitat names appear only in the
  `tp` cluster (112–424 mentions each, from legitimate template
  utterances like "Welche Tiere leben in der Hartholzaue?"). Regenerating
  vectors is out of reach — no in-repo build script exists
  (`botVectors.py` only loads; embeddings come from the external bge-m3
  endpoint).
- Stage 2 of `_match_input_to_items_structured` (per-token WRatio
  fallback, threshold 65, tokens ≥ 5 chars) accepts lookalikes:
  `WRatio("feuchtwies", "efeu") = 77.1`, `WRatio("luchs", "rotfuchs") =
  80.0`, while a genuine typo scores 90.9 (`feuchtwies` →
  `feuchtwiesen`).
- Downstream is already compatible: `find_entity` with `type='TP'`
  searches all 170 rawData entries and infers Typ from the top match, so
  an entity requirement filled with an Auen name resolves to real
  content (Hartholzaue: Erkennungsmerkmale 1233 chars).
- The LLM-arbiter pattern exists: `botDecoder.detectIntent` already calls
  `OpenAICompatClient.chat_json` (temperature 0,
  `system_prompt_check_intent_de`) to break intent-option ties.

## Goals / Non-Goals

**Goals:**

- Auen-name queries resolve to real entity content through the existing
  complete-handler pipeline.
- The fuzzy fallback stops inventing entities for unrelated tokens.
- No changes to data files, vector regeneration, API contract, or
  storage.

**Non-Goals:**

- Regenerating `intent_vectors.json` or adding a vector build script.
- Routing Auen queries to the `aue_*` intent family (their `default`
  handler cannot resolve entities; would swap an error for generic
  text).
- Merging Auen names into `entity_tp.json`.
- LLM arbitration on successful match paths (only failure paths).
- Auto-picking among *intent* options — the ask-the-user options flow
  stays as designed.

## Decisions

### D1: Arbiter as failure-path hook in the complete handler

`_handle_no_options` calls the arbiter exactly when
`_match_input_to_items_structured` returns `None` (which after D2
includes stage-2 rejections). On success it fills the `entity`
requirement from the arbiter's validated answer and continues the
normal pipeline; on any failure it falls through to the existing
`_make_error_response("no_match", ...)` path. Rationale: zero behavior
change on the success paths, one chat call only where today an error is
returned anyway.

### D2: Stage-2 shared-prefix acceptance

In stage 2, a fuzzy candidate is only accepted when the first 3
characters of the token stem equal the first 3 characters of the
matched name's stem (or the token stem is contained in / contained by
the name stem at the start). Verified effect: `feuchtwies→efeu` (77.1)
and `luchs→rotfuchs` (80.0) rejected; `feuchtwies→feuchtwiesen` (90.9)
accepted. The prefix is a stem-level check, so umlaut mapping applies
consistently on both sides.

### D3: Candidate lists from the three entity files, strict validation

The arbiter prompt embeds the name lists loaded from
`entity_tiere.json`, `entity_pflanzen.json`, `entity_aue.json` (98 + 64
+ 8 names, ~2 KB of text — well within one chat call). The LLM returns
JSON `{"domain": "Tier"|"Pflanze"|"Auen", "name": "...", "found": bool}`
(`chat_json` with a schema hint, temperature 0). The answer is valid
only if `name` matches exactly one entry of the claimed domain's file
(case-insensitive, after whitespace trim); otherwise — including
`found: false`, malformed JSON, unknown domain, or request failure — the
arbiter reports "not found" and the error path proceeds. Rationale:
hallucination cannot enter the pipeline; `entity_aue.json` stays the
single owner of the Auen names.

### D4: Arbiter also handles in-sentence terms

The prompt instructs the model to pick the domain entity name that the
user's input refers to, including when it appears inside a longer
sentence ("was ist die Feuchtwiese im NSG" → `Feuchtwiesen`). Same
single call; validation unchanged. Multi-word and umlaut-stem variants
are matched case-insensitively against the lists.

### D5: `llm_used` observability

When the arbiter runs (regardless of outcome), the flag propagates so
the `history.llm_used` column reflects it, matching how
`detectIntent`'s LLM tie-breaker reports usage. The history-staging
behavior of `store_history` is otherwise untouched.

### D6: Verification through the existing harness

`botApiTest.py` extends its `entities` sample list (and keeps all
existing scenarios):

| new sample | expectation |
|---|---|
| `hartholzaue` | entity resolves to Hartholzaue; text carries real Erkennungsmerkmale content |
| `stillgewässer` | resolves to Stillgewässer |
| `feuchtwiesen` | resolves to Feuchtwiesen (was: Efeu) |
| `luchs` | no species resolution — no-match/options path, never an animal |
| `magerrasen` | resolves to Magerrasen |

The existing samples (`reh`, frog groups, `windröschen` lookalike
protection, `knabenkraut` pick flow, `rheinaue`, gibberish no-match)
MUST stay green unchanged; the `frosch` scenario and the token/security
suite are regression guards.

### D7: Arbiter recovery normalizes the feature to 'definition'

A recovered entity usually belongs to a different domain than the
decoded intent (an Auen name under a `tp_*` intent), so the
intent-provided feature often has no data for it — `tp_generell`'s
`feature: 'generell'` maps to a `"Generell"` field no entity has, so
even a correctly resolved entity answered "keine Informationen". On
arbiter recovery the feature is therefore set to `definition`, which
maps to the full key list (`self.keys`) and yields real content for
every entity. Decided during apply (user choice); resolves the open
question below.

## Risks / Trade-offs

- [LLM latency on the failure path] → One call (temperature 0, small
  prompt) only where today an immediate error was returned; worst case
  adds the LLM timeout (30 s) to an already-failing turn. No effect on
  successful paths.
- [LLM misclassification within the lists] → Strict exact-match
  validation (D3) means the worst case is a *wrong but real* entity if
  the model names a valid one; the domain must also match its file, so
  cross-domain confusions are rejected. Accepted residual risk, same
  class as the existing intent tie-breaker.
- [Prompt size grows with the lists] → ~170 names / ~2 KB; bounded by
  the static data files. If the data grows substantially, revisit with
  per-domain prompts.
- [arbiter masks genuine coverage gaps] → The no-match path is now
  rarer, so missing data hides behind LLM guesses only when the name
  truly exists in a list; names absent from all three files still fail.
  History rows with `llm_used=true` make arbiter-routed turns auditable.
- [stage-2 prefix check rejects typos with different first letters] →
  Deliberate tradeoff: a typo like "orcfrosch" for "Moorfrosch" no
  longer fuzzy-matches, but the arbiter covers exactly this case
  (input fails stage 2 → LLM recovers `Moorfrosch` from the Tier list).

## Migration Plan

1. Implement D2 (stage-2 prefix check); run the existing suite — the
   `entities` samples for calibrated fuzzy cases must stay green.
2. Implement D1/D3/D4/D5 (arbiter + hook); add the D6 samples; run the
   full `--scenario all` suite green.
3. Rollback: revert the branch — no data or schema changes to undo.

## Open Questions

- (resolved during apply — see D7) Which Auen feature the user expects
  for a bare "hartholzaue" query: arbiter recovery now normalizes the
  feature to `definition` (full key list). Habitat-specific feature
  mapping remains a possible follow-up intent-modeling change.
