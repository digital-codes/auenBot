# Proposal

## Why

Since the sealed-state change, user-facing option lists (intent
choices, species group matches, feature-fallback alternatives) travel
only inside the encrypted `state` token. The frontend cannot decrypt,
so it can no longer render option buttons — the bot says "Bitte wählen
Sie…" and the choices are invisible. The exploration of the
`auenlaend` frontend (`ChatPane.vue`) confirmed the option rendering
reads `message.context.options`, a field that no longer exists on the
wire. The choices themselves are not secret — the prompt text already
announces them — so they can safely travel in plaintext next to the
sealed token, while the actual conversation state (intent, entity,
feature) stays sealed.

## What Changes

- Both API response sites in `route_handler` (the option-prompt early
  return and the final intent-execution response) include an `options`
  field with the current option list whenever the context carries
  options; responses without options omit the field.
- The shape is exactly the context's option list (title dicts), so the
  frontend's existing option rendering keeps working unchanged.
- Options remain sealed inside the `state` token as well — the seal
  input is unchanged, and no other context field is exposed.
- `botApiTest.py` gains assertions that option-prompt responses carry
  the plaintext options alongside the sealed token (frosch group
  scenario is the natural place).
- Frontend work (`auenlaend`): switch the response gate and request
  echo from `context` to `state` and render `options` from the new
  field — **out of scope here** (separate repository), tracked as the
  consumer of this contract.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `state-token-encryption`: Add the requirement that user-facing
  option lists accompany the sealed state in plaintext; the seal
  itself and the protection of intent/entity/feature are unchanged.
- `bot-api-testing`: Add the requirement that option-prompt responses
  carry the plaintext option list alongside the sealed token.

## Impact

- **Code**: `botCtl.py` only — the two response sites in
  `route_handler`. `botApiTest.py` (option assertions).
- **API contract**: additive and non-breaking — responses gain an
  optional `options` field; existing consumers (the harness) ignore
  unknown fields. The request contract is unchanged.
- **Security**: no new exposure of protected state — only the
  user-visible choices; tokens stay tamper-evident and opaque for
  intent/entity/feature.
- **Not affected**: sealing logic, sanitization, arbiter, focus/
  breakout, `bot_history.db`, the request schema.
