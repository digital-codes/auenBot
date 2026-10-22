# Design

## Context

Both mechanisms were explored against the current code (see the
fork-comparison discussion): the completion flow leaves a user stuck in
an infinite *"Nichts gefunden"* loop because `_make_error_response`
keeps `ctx.intent` set (it only pops `options`/`target`), so every
following input takes the "intent already set" branch back into the
same completion; and the schema's `last_input`/`last_intent` fields are
read in `route_handler` (lines ~472–478) but never written — the writes
are commented out in `_finalize_completion` and `_make_error_response`.
The fork's `QaEngine` derives retry counts by scanning an event log
backwards; the `qAndA` prototype confirms follow-ups against a `focus`
snapshot. Both ideas map onto odd26's stateless sealed-context
round-trip without any wire change: new context keys travel inside the
sealed `state` token exactly like `intent`/`entity` do.

## Goals / Non-Goals

**Goals:**

- An exit from repeated misunderstanding: a full conversation reset
  after a configurable number of consecutive misses.
- Natural follow-ups ("und wie groß ist sie?") answered from the last
  completed entity.
- Zero wire/schema changes; new keys ride the existing sealed state.

**Non-Goals:**

- The fork's full FSM/`HistoryEvent` machinery (declarative transitions,
  per-event sealing) — this change takes only the two mechanisms with
  the best value/cost; the FSM remains a possible follow-up.
- Confirmation turns for follow-ups (qAndA-style "Do you mean ...?"") —
  direct fill chosen (D3); revisit if misfires appear.
- Anaphora/NLU resolution ("sie"/"er" pronoun handling) — the trigger is
  "feature intent + unresolvable entity + focus exists", not linguistic
  analysis.
- Clearing the session id or sequence numbering on breakout.

## Decisions

### D1: Miss counting lives centrally in `route_handler`

`intents.execute()` returns `error` on the no-match paths (both
`_handle_no_options` and `_handle_with_options` funnel there). After
the call, `route_handler` inspects the result: on a no-match error it
increments `ctx["no_match_count"]`; on any successful match (no error)
it resets it to 0. Rationale: one hook covers both existing no-match
sites and any future one; `botIntents.py` stays free of counter logic;
the count is sealed into the response token exactly like other context
keys. The detection key is the error outcome, not message text.
Threshold check happens in the same hook before store_history/sealing.

### D2: Breakout = context wipe with an inviting reply

At the threshold, `route_handler` replaces the working context with an
empty dict and substitutes the reply text with a restart invitation
("Ich hab dich leider nicht verstanden. Lass uns von vorne beginnen —
stell deine Frage einfach neu!"). Empty context means the next input
decodes fresh — identical to a brand-new conversation, but the session
id and sequence continue, so history rows stay grouped. The decline
word list in `route_handler` is untouched and still works
independently.

### D3: Follow-ups fill from the focus, gated by an LLM follow-up check

In `_handle_no_options`, when the requirement is `entity`, staged
matching returned nothing, the arbiter returned nothing, AND
`context["focus"]["entity"]` exists, the entity requirement is filled
from the focus — but only after `followup_check` (one `chat_json`
call, same pattern as the existing LLM arbiters) confirms the input
refers to the focused entity. Rationale from apply-turn verification:
without the gate, gibberish after an answer is silently answered with
the focused entity's content (the sticky intent re-enters the
completion flow forever and the breakout becomes unreachable), while
the genuine follow-up must still get through. A "no" or an LLM failure
stays on the no-match path — degradation preserves the breakout.
Direct fill (no confirmation turn) is kept for confirmed follow-ups;
the bio_feature reply names the entity, making the fill transparent.
The intent-provided feature is kept (the follow-up names a concrete
aspect); when it has no data, D8's feature fallback takes over.

### D4: Focus snapshot written in `_finalize_completion`

On every successful completion, `_finalize_completion` sets
`ctx["focus"] = {"entity": <resolved entity>, "type": <type>}` (and
`ctx["last_input"]`/`ctx["last_intent"]` per D5). Each completion
replaces the focus — so follow-ups always target the most recent
answer. The group-match/options path does not set a focus (nothing was
answered yet).

### D5: `last_input`/`last_intent` become live in the same place

The commented-out writes in `_finalize_completion` are implemented
there (`last_input = input_text`, `last_intent = intent name`),
matching the fields the old schema documented and `route_handler`
already reads. No new fields — the names survive in history plaintext
and the sealed token.

### D6: No-match counter and focus are plain context keys

No special serialization: `no_match_count` (int), `focus` (dict),
`last_input`/`last_intent` (str) ride the sealed token like every other
key, appear in the plaintext `bot_history.db` rows, and are cleared by
the breakout wipe. `_make_error_response`'s copy-and-pop behavior
leaves them intact (it only pops `options`/`target`).

### D8: Empty feature lookups offer available features

When a resolved entity has no data for the requested feature (the
*"Zu dieser Eigenschaft ... leider keine Informationen"* case — e.g.
Reh has no `Größe` field, or `tp_generell`'s `generell` key), the
completion no longer dead-ends: `get_entity_features` marks the result
(`no_feature_data`) and reports up to 3 other fields of that entity
that have data (meta keys like `Name*`/`Typ`/`Gruppe`/`Links`
excluded). `_finalize_completion` intercepts the marked result, keeps
`entity` in the context, **pops `feature`**, sets the offered features
as context options, and replies with an informing prompt. The next
reply is matched against those options by the existing
`_handle_with_options` flow — picking one fills the feature requirement
and the pipeline answers with that feature's content (any field name
works as a feature key via the mapping's passthrough branch). Decided
during apply (user choice): this applies to all empty feature lookups,
not only focus-filled turns. If the entity has no other data-carrying
fields, the existing no-information reply is kept.

### D9: Weak group matches defer to the focus; loose containment needs 4-char stems

Applied-turn verification exposed two matcher interactions: (a) 3-char
loose stems like `gro` prefix-match dozens of `Groß-*` entities at once,
and (b) even the exact word `groß` rank-2-prefix-matches `Großer-*`/
`Große-*` names, so a follow-up like *"und wie groß ist sie?"* hijacked
the turn as a bogus 10-animal group instead of reaching the focus fill.
Fixes: loose rank-2 containment requires token stems of 4+ characters
(D9a), and a group match that rests **only** on rank-2 loose hits is
marked *weak* — when a focus exists and the entity requirement is
unresolved, a weak group defers to the focus fill instead of offering
its options (D9b). Strong matches (rank 0/1 exact stem relations, e.g.
*frösche*) are unaffected. Without a focus, weak groups keep the
existing options behavior.

### D10: Only unhelpful focus-filled turns count as misses

The deferred D3 question is decided by observed behavior: a focus-filled
follow-up that **answers with content** is helpful and resets the miss
counter (so genuine follow-up chains never break out); a focus-filled
turn that falls into the D8 feature-fallback (the feature has no data —
the answer was unhelpful) counts as a miss. Sequence check: gibberish
after an answer alternates focus-fill-fallback (count) / option-mismatch
(count) → breakout at the limit; genuine follow-up chains always reset.
The flags travel on the result (`focus_fill`, `no_feature_data`).

## Risks / Trade-offs

- [Direct focus fill answers a stale entity on decoder misfires] → The
  trigger requires a tp/tiere/pflanzen feature intent AND failed
  matching AND failed arbiter; smalltalk and options turns never reach
  it. Residual risk: a gibberish input decoded into a tp intent gets a
  real (if unexpected) answer instead of the error — accepted as more
  helpful than an error, and the miss counter still tracks the turn
  via... (note: a focus-filled turn is a successful match and resets
  the counter; if this proves too permissive, the fill can mark the
  turn `no_match`-like for counting — decide by observing behavior).
- [Breakout wipes a context the user could have recovered from] → The
  alternative (looping errors) is worse; the threshold is configurable
  and the reply explicitly invites re-asking.
- [Counter counts globally, not per intent] → Consecutive misses
  across different tp intents still escalate — desired: the user is
  visibly stuck regardless of which intent decoded.
- [`no_match_count` leaks into decrypted wire context] → Harmless: the
  harness decrypts anyway; the client cannot forge anything new (the
  token is sealed; tampering fails the auth tag).

## Migration Plan

1. Implement D1/D2 (counter + breakout in `route_handler`, threshold
   setter); add the breakout scenario to the harness.
2. Implement D3/D4/D5 (focus + trace writes); add the follow-up
   scenario.
3. Run the full `--scenario all` suite green (existing 15 samples and
   security/restart scenarios as regression guards).
4. Rollback: revert the branch — no data, schema, or wire changes.

## Open Questions

- (resolved during apply — see D10) Whether a focus-filled follow-up
  turn counts toward the miss counter: only when it could not answer
  (feature fallback fired); content-answering follow-ups reset.
