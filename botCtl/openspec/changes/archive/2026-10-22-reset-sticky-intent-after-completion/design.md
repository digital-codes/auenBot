# Design

## Context

`route_handler` in `botCtl.py` implements the per-turn routing. The
context arrives as a sealed token; `ctx["intent"]` currently acts as an
unbounded "sticky" routing token: any turn whose context carries
`intent` skips decoding entirely (`elif user_intent is not None ...`).
Completion-style intents (`handler: complete`, e.g. `messdaten`,
`anreise`) set `ctx["intent"]` whenever they present options or need
more turns, and `_finalize_completion` removes the requirements and
options but leaves `intent` in place. That combination routes every
later turn into the last completed intent — see proposal for the
observed history.

Also relevant: `_handle_no_options` re-presents an intent's option file
on every execution that arrives without options in context; combined
with the sticky intent this recreates the menu forever, and each
re-presentation resets the miss counter, so the existing breakout never
fires.

## Goals / Non-Goals

**Goals:**
- After a completion answers, the next turn is decoded fresh; other
  intents become reachable again within the same session.
- Focus-based follow-ups (`last_intent`/`last_input`/`focus`) keep
  working unchanged.
- A pending flow still consumes option picks (multi-turn completions
  like anreise/messdaten keep working).
- A pending flow that cannot use the input hands the turn to a newly
  decoded intent instead of erroring out with internal text.

**Non-Goals:**
- No change to the sealed-state format, seal/unseal, or wire contract
  beyond `intent` disappearing from completed states.
- No redesign of the decoder (`BotDecoder`) or of the entity
  arbiter/focus mechanics.
- No frontend work.

## Decisions

**D1 — `intent` becomes a pending-completion marker; pop it on
completion.**
`_finalize_completion` (and the target-routing branch inside it) pops
`intent` from the returned context. Requirements, options, `last_*`
and `focus` handling stay as today. Alternative considered: keep
`intent` and add a `completed` flag the router checks — rejected
because the sealed state already leaks today's shape to every future
turn and two sources of truth (`intent` + flag) recreate the same
desync that caused this bug. A single marker whose lifecycle is "set
while waiting, gone when done" is simpler and self-healing.

**D2 — Routing precedence in `route_handler` becomes:
decline/reset > option pick > pending intent > fresh decode.**
Concretely:
1. Decline/reset keywords → decline intent (unchanged).
2. If the decoder's intent options are pending (no completion intent
   attached), option resolution runs first; a match consumes them.
   Completion menus (pending completion intent) are NOT routed through
   option resolution: their option titles are values ("O3"), not
   intent names — the pending intent consumes the pick itself in
   `_handle_with_options`.
3. If a pending intent exists and no option matched, execute the
   pending intent once (it may fill requirements from the input text,
   as tp_* flows do today).
4. Only when that execution yields a no-match, fall back to
   `decoder.detectIntent(user_input)`; a decisively decoded, different
   intent replaces the pending one for this turn. An unsure re-decode
   (fallback intent or only option candidates) keeps the no-match
   reply and counts the turn as a miss.

Alternative considered: always re-decode first and only use the pending
intent when decoding agrees — rejected: it would break pending flows
whose continuation text ("O3", "Erkennungsmerkmale") does not decode to
the intent itself, and costs an LLM call on every option pick.

**D2a — Only answering turns reset the miss streak.** Two turn types
end without content and now count as misses so repeated gibberish can
actually reach the breakout: (a) a turn answered by the fallback
"not sure" utterance after a failed fresh decode, and (b) an
intent-options prompt offered mid-conversation — when the context
already carries a focus or an active streak. The prompt preserves and
increments the counter instead of resetting it; picking an offered
option afterwards answers and resets. The first ambiguous turn in a
clean session still prompts options without counting a miss. This
replaces the miss counting the old sticky routing incidentally
provided (gibberish was forced into the stale intent and failed entity
resolution there).

**D3 — The no-match fallback lives in the router, not in
BotIntent.** `execute()` returns a `no_match` marker (it already does
via `result["no_match"]`). The router reacts to it by re-decoding and,
on success, re-entering `intents.execute` with the new intent. This
keeps the miss counting, breakout and history storage in one place
(the router already owns them) instead of duplicating the loop inside
the handler.

**D4 — User-facing unmatched-input text.** The two observed leak sites
are `_make_error_response` calls that pass internal keys or ad-hoc
placeholder strings as `msg_key`. The no-match paths get one shared,
localized user sentence (e.g. "Das habe ich leider nicht verstanden —
stell deine Frage einfach neu."). `no_match` marking and miss counting
are unaffected; only the outgoing text changes. Internal keys stay
available for logging via the existing `status`/history columns.

**D5 — `type`/`feature` residuals are left to the requirements
cleanup.** `_finalize_completion` already pops requirement keys;
`type` for messdaten (`requires: type`) is therefore removed on
completion. No extra scrubbing pass is added — a completion that
leaves a stray scalar cannot route turns anymore, because routing keys
off `intent`/`options` only.

**D6 — Default-handler intents clear the marker too.** The default
handler answers immediately but returns its context unchanged, so the
marker the router attached before execution would survive the turn and
stickiness would simply move to the last default intent. The default
handler therefore pops `intent` like `_finalize_completion` does.

## Risks / Trade-offs

- [Re-decode on no-match costs an extra LLM call on a failing turn]
  → only on turns that already failed; gibberish turns hit the local
  fallback intent path without LLM intent calls beyond the existing
  detection call.
- [Decoder misreads an option-pick continuation as a different intent]
  → option resolution runs before the pending intent is consulted
  (D2), so exact/title picks never reach the decoder; fuzzy fallback
  only after the pending execution already reported no-match.
- [States sealed by the old service (intent set forever) replayed
  after deploy] → a stale `intent` now routes to D2 step 3 once; if
  that execution completes, the marker is popped on the very next
  seal. Worst case is one recycled answer, not a loop.
- [Tests assert on today's sticky behavior] → `botApiTest.py` scenarios
  are updated in this change; the post-completion assertions are the
  new spec's acceptance criteria.

## Migration Plan

Single-service deploy; no DB migration (history schema unchanged, the
`context` column is free-form). Rollback = redeploy previous version;
sealed tokens minted by the new code simply carry `intent` again.

## Open Questions

None.
