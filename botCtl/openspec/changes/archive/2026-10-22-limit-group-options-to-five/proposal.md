# Proposal

## Why

Large group matches overwhelm the species pick prompt: a query like
*"wie groß ist das reh"* loosely matches the exact entity plus every
`Groß-*` species, and the completion flow offers **all** of them —
11+ option buttons in the frontend (observed during the frontend
contract exploration). Five choices is the usable maximum for a
pick prompt (and matches the largest legitimate group, the five
frosch species). The cap belongs in the backend, which owns the
option list; the frontend renders whatever it receives, per contract.

## What Changes

- Group matches in the entity matcher are capped at **5** offered
  options: when the match ranking distinguishes the members (mixed
  ranks), the best-ranked five are offered in rank order; when all
  members share the same rank (no ranking signal), five are chosen at
  random so repeated asks can surface different members.
- Groups with five or fewer members (e.g. the five frosch species,
  the two Windröschen) are unchanged — full list, same order.
- The plaintext `options` exposure and the sealed state automatically
  carry the capped list (they project the context's option list).
- `botApiTest.py` gains a capped-group sample: a query producing a
  large group must offer at most 5 options.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `entity-resolution`: The group-option requirement changes from
  "present all matching entity items" to "present at most five",
  with the best-ranked-or-random selection rule.

## Impact

- **Code**: `botIntents.py` `_match_input_to_items_structured` (cap
  applied where the group names and their ranks are both known).
  The weak-group demotion, focus fill, and feature fallback are
  unaffected.
- **Behavior**: only groups larger than five change — the offered
  list shrinks; the user can still type an unlisted species name
  directly (stage-1 matching resolves it on the next turn).
- **Not affected**: single matches, small groups (≤ 5), frosch/
  knabenkraut/fledermaus scenarios, the plaintext options exposure,
  the sealed token, breakout/follow-up semantics.
