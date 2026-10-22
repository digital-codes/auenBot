# Design

## Context

The exploration of the `auenlaend` frontend (`ChatPane.vue`,
`BotResponse.vue`, `backendComms.ts`) showed the option rendering reads
`message.context.options` — a field removed from the wire by the
sealed-state change. Options now travel only inside the encrypted
token, so intent choices, species groups, and feature-fallback
alternatives are invisible to the client. The choices are already
announced by the prompt text and carry no protected state, so exposing
them in plaintext restores the UI without weakening the seal (intent,
entity, feature, focus stay token-only).

## Goals / Non-Goals

**Goals:**

- Option-prompt responses carry the plaintext option list next to the
  sealed token, in the exact shape the frontend already renders.
- The seal input is unchanged; options stay inside the token too.
- Zero request-side changes; fully additive response field.

**Non-Goals:**

- Frontend changes (`auenlaend` repo): switching the response gate and
  request echo from `context` to `state` is the consumer's work and is
  tracked there.
- Exposing any other context field (intent, entity, feature, focus).
- A separate response schema file (the request schema is unaffected;
  the response field is documented by the spec).

## Decisions

### D1: Expose at the two response sites in `route_handler`

Both `jsonify` sites (the option-prompt early return and the final
intent-execution response) include `"options": ctx["options"]` when the
context carries a non-empty option list, and omit the key otherwise.
Rationale: both option flows (group matches via `_handle_no_options`,
feature fallback via `_finalize_completion`, and intent options via the
decoder) surface through these two sites; a helper is unnecessary for
two sites, and the condition keeps non-option responses byte-identical
to today.

### D2: Same shape, no transformation

The plaintext list is the context's option entries verbatim — title
dicts, possibly with `text` aliases for intent options. Rationale: the
frontend's `BotResponse.vue` renders `{title, text?}` today; a
transformation would add a translation layer with no consumer need.

### D3: Options stay inside the seal

`seal_state(ctx, session)` is called with the unchanged context,
options included. Rationale: the token remains the single
authoritative state (tamper-evident, session-bound); the plaintext
list is a display projection, and the D8/option flows verify state
through the token, not the plaintext copy.

### D4: Harness assertions on the frosch scenario

The frosch scenario (five-species group) is the deterministic
option-prompt case: assert the response carries `options` with exactly
the five titles next to the token, and that a reh-style option-less
response omits the field. Rationale: deterministic, already green, and
exercises both branches of the condition.

## Risks / Trade-offs

- [Options are client-visible] → Intended: they are the prompt's
  choices, already announced by the message text. Protected fields
  (intent/entity/feature/focus) stay sealed; the harness asserts the
  token still hides them.
- [Response payloads grow slightly on option turns] → Bounded by the
  option lists (≤ 10 entries typically; D8 caps at 3).
- [Two response sites to keep in sync] → Acceptable; a response helper
  would be the refactor if a third site appears.

## Migration Plan

1. Add the `options` field at both response sites; run the frosch and
   entities scenarios — new assertions green, everything else
   unchanged.
2. Frontend (`auenlaend`, separate repo): switch the response gate and
   request echo from `context` to `state`, render `message.options`.
   Lockstep with this change restores option buttons.
3. Rollback: revert the branch — additive field, no data or state
   impact.

## Open Questions

(none)
