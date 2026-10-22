# Tasks

## 1. No-match breakout

- [x] 1.1 Add the miss-count hook in `botCtl.py` `route_handler`: detect the no-match error outcome of `intents.execute()`, increment `ctx["no_match_count"]` on it, reset it to 0 on any successful match, and add a `setMaxNoMatch(n)` setter (default 3) wired at startup next to `setThreshold(65)`; verify by direct code inspection plus a live two-turn curl check that the decrypted context carries `no_match_count` increasing across consecutive unresolvable inputs
- [x] 1.2 Implement the breakout: when the count reaches the configured limit, replace the working context with an empty dict and the reply with a restart invitation before store_history and sealing; verify end-to-end: three consecutive gibberish inputs against one session produce the invitation on the third turn and a decrypted context of `{}` afterward
- [x] 1.3 Verify counter semantics: a successful match between misses resets the count (breakout does not trigger at the limit), the configurable limit is honored (set to 2 via the setter, breakout on the second miss), and the decline-word reset still works independently

## 2. Focus-based follow-ups

- [x] 2.1 In `botIntents.py` `_finalize_completion`, set `ctx["focus"] = {"entity": <resolved>, "type": <type>}` plus `ctx["last_input"]` and `ctx["last_intent"]` on successful completion (group/options paths set no focus); verify by live turn: after "was macht das reh" the decrypted context carries `focus.entity = "Reh"`, `last_input`, and `last_intent`
- [x] 2.2 Add the focus fill to `_handle_no_options`: when the requirement is `entity`, staged matching returned nothing, the arbiter returned nothing, and `context["focus"]["entity"]` exists, fill the entity from the focus and continue the pipeline with the newly decoded intent's feature; verify end-to-end: "was macht das reh" followed by "und wie groß ist sie?" returns Reh size content (not "Nichts gefunden"), and the second completion's focus points at the new entity
- [x] 2.3 Verify follow-up containment: "moorfrosch" still resolves via normal matching (focus not consulted), a breakout reset clears the focus (no follow-up after reset), and smalltalk/option turns are unaffected

- [x] 2.4 Implement the feature-fallback options (design D8): `get_entity_features` marks empty lookups (`no_feature_data`) and reports up to 3 data-carrying fields (meta keys excluded); `_finalize_completion` intercepts, keeps `entity`, pops `feature`, sets the fields as context options, and replies with an informing prompt; verify end-to-end: "und wie groß ist sie?" after the Reh answer offers options, picking one returns real content, and an entity with no data at all keeps the existing no-information reply

- [x] 2.5 Matcher refinements required by verification (design D9): loose rank-2 containment demands 4-char stems; rank-1 hits distinguish the compound noun head (final word: "frosch" in "Teichfrosch" = strong) from adjective lead words ("groß" in "Großer ..." = weak); weak-only group matches defer to the focus; verify the calibrated matcher cases stay green and the follow-up query is marked weak
- [x] 2.6 LLM follow-up gate (design D3, decided during apply): the focus fill fires only after `followup_check` confirms the input refers to the focused entity (same chat_json pattern); a "no" or LLM failure stays on the no-match path; verify gibberish after an answer counts toward the breakout while the genuine follow-up answers with content

## 3. Harness regression scenarios

- [x] 3.1 Add a breakout scenario to `botApiTest.py` (consecutive unresolvable inputs until the limit, expect the invitation reply and an empty decrypted context) and a follow-up scenario (reh answer, then "und wie groß ist sie?": expect the informed fallback with up to 3 feature options since Reh lacks Größe data, then pick one option and expect real content); verify both PASS against the running service
- [x] 3.2 Verify the full suite stays green: `python botApiTest.py --scenario all` exits 0 with all scenarios PASS (reh, frosch, the 15 entity samples, breakout, follow-up, tampered, foreign, db_crosscheck, restart_resume, restart_no_secret)

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Deferred design note: whether a focus-filled follow-up turn should count toward the miss counter — decide by observation (design D3 risk note).
- Deferred follow-up (fork comparison): full FSM/HistoryEvent machinery and per-event sealing remain a possible later change.
