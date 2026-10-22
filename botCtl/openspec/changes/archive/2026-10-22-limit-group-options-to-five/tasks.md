# Tasks

## 1. Group cap in the matcher

- [x] 1.1 In `botIntents.py` `_match_input_to_items_structured`: cap group results at 5 names — when the group's members span multiple ranks, keep the first five of the rank-ordered list; when all share one rank, pick five at random (`random.sample`); groups of ≤ 5 pass through unchanged and the `weak` flag is computed before the cap; verify by direct call: an oversized loose group (e.g. "und wie groß ist sie?" vs `entity_tp.json`) returns exactly 5 names while "gibt es da frösche" still returns its 5, "knabenkraut" its 2, and "maus" its 3
- [x] 1.2 Verify end-to-end via the running service: "wie groß ist das reh" offers at most 5 options with `Reh` among them (rank ordering), and the plaintext `options` field on the wire carries the same capped list

## 2. Harness sample and regression

- [x] 2.1 Add a capped-group sample to `botApiTest.py` `ENTITY_SAMPLES`: "wie groß ist das reh" expects options to be offered with at most 5 entries (add a `max_options` expectation to the sample runner); verify PASS against the running service
- [x] 2.2 Verify the full suite stays green: `python botApiTest.py --scenario all` exits 0 with all scenarios PASS (the frosch, knabenkraut, and fledermäuse group scenarios are unchanged; all 15 entity samples, breakout, follow-up, token security and restart scenarios, DB crosscheck)

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
