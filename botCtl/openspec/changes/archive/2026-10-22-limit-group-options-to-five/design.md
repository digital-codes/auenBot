# Design

## Context

Group matches in `_match_input_to_items_structured` rank hits by
quality (rank 0 exact stem, rank 1 compound noun head, rank 2
loose/lead-adjective) and return the deduplicated, best-rank-first
name list — unbounded. The completion flow offers every name as an
option, so loose queries produce oversized menus (11+ buttons
observed for *"wie groß ist das reh"*: the exact `Reh` hit plus every
`Groß-*` species). The frontend renders the list verbatim, per
contract. The matcher is the one place where the ranks and the names
coexist, so the cap belongs there.

## Goals / Non-Goals

**Goals:**

- Species pick prompts offer at most five options.
- Ranking (when informative) decides which five; variety (random)
  when it does not.
- Small groups and all existing calibrated scenarios unchanged.

**Non-Goals:**

- Capping intent options from the vector decoder (a different flow,
  already de-duplicated and user-confirmed).
- Capping the D8 feature fallback (already capped at 3).
- Pagination or "more…" options in the pick prompt.
- Changing the plaintext options exposure or the seal.

## Decisions

### D1: Cap inside the matcher, where ranks live

`_match_input_to_items_structured` applies the cap when building the
group result: it knows each name's best rank. Groups of ≤ 5 pass
through unchanged. The `weak` flag logic is unaffected (it is
computed from `min_rank` before the cap).

### D2: Best-ranked five when ranks differ, random five otherwise

If the group's members span more than one rank (the ranking carries
information — e.g. the exact `Reh` hit vs. loose `Groß-*` hits), the
first five of the rank-ordered list are offered: the strongest
matches stay visible, which for mixed queries keeps the user's
actually-named entity in the list. If all members share one rank
(no signal — e.g. a bare stem loosely prefixing a uniform family),
`random.sample` picks five, so repeated asks explore the family.
Rationale: deterministic quality ordering where it exists, honest
variety where it does not; both rules satisfy "best 5 if score
available else random 5".

### D3: Cap at the group result, not the options prompt

The returned `match_result["names"]` is capped, so every consumer
(the options prompt, the plaintext exposure, the sealed state) sees
the capped list without extra plumbing. The demotion/focus logic
upstream of the group branch is unaffected. Risk noted: a capped
list can omit the user's intended species — they can type it
directly; stage-1 matching resolves it on the next turn.

## Risks / Trade-offs

- [Intended entity not among the offered five] → Mitigated by D2's
  rank ordering (exact/noun-head hits rank first) and by direct
  typing; the group prompt asks to choose from the options but the
  matcher still resolves any named species.
- [Random selection makes assertions non-deterministic] → Only for
  uniform-rank groups larger than five; the harness asserts the cap
  (≤ 5), not specific members. Deterministic cases (frosch = 5,
  knabenkraut = 2) keep exact assertions.
- [Existing spec said "all matching entity items"] → This change
  MODIFIES that requirement deliberately (delta spec); the five
  frosch scenario is unchanged and stays green.

## Migration Plan

1. Implement D1/D2 in the matcher; verify the calibrated matcher
   cases (frosch group of 5 unchanged, knabenkraut/fledermaus
   unchanged) and the cap on an oversized group.
2. Add the capped-group harness sample ("wie groß ist das reh" →
   at most 5 options, Reh among them when ranks differ).
3. Run the full suite green.
4. Rollback: revert the branch — purely behavioral, no data or
   contract changes.

## Open Questions

(none)
