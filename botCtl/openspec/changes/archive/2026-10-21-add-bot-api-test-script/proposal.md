# Proposal

## Why

The `botCtl.py` Flask service processes user input through a chain of
intent detection, entity resolution, and feature lookup against local
JSON/vector data. During manual runs (see `botCtl.log`) the intent and
entity are resolved correctly for the input *"was macht das Reh"*
(intent `tp_definition`, entity `Reh`, confidence 90%/100%), but the
feature lookup returns content that mixes descriptions from unrelated
entities (e.g. *Köcherfliegen*, *Regenbremse*) together with the actual
*Reh* record. The root cause is suspected to be incorrect handling of
fuzzy-match indices / sorting inside `BotAction.find_entity`
(`botActions.py`), but there is no executable way to reproduce this
against the running service today.

We need an HTTP-driven test script that drives a live `botCtl`
instance, captures the conversation context per request, and can be
extended with more scenarios over time. The first scenario mirrors the
*Reh* sample from `botCtl.log` so the suspected bio-feature bug can be
characterized and reproduced reliably. Fixing the underlying bug is
**out of scope** for this change and will be addressed in a follow-up
change once the test makes the failure observable.

## What Changes

- Add a new Python script `botApiTest.py` at the repository root that:
  - Sends JSON requests to the `POST /api` endpoint exposed by a
    running `botCtl.py` instance (default `http://127.0.0.1:11354/api`).
  - Reuses the request schema defined in `botSchema.json`
    (`input`, `session`, `sequence`, `lang`, `message`, `context`).
  - Persists and re-supplies `session`, `sequence`, and `context`
    between calls so the bot's state-machine is exercised end-to-end.
  - Prints or returns the response `message.text`, `context`, and
    `error` fields so a human or CI runner can assert on them.
- Add a first concrete scenario in the script that reproduces the
  *Reh* case from `botCtl.log`: POST `"was macht das reh"` (German,
  fresh session) and print the resulting `context.entity`,
  `context.feature`, and `message.text`. The scenario does **not**
  assert on the bio-feature text yet — it only asserts on
  intent/entity resolution so we can confirm the reproduction step
  before adding bug-specific assertions in a later change.
- No existing files are modified. No production code in
  `botCtl.py`/`botActions.py`/`botIntents.py` changes in this change.
- Subsequent changes (separate proposals) will:
  - Add more scenarios.
  - Assert on the bio-feature output.
  - Fix the suspected `find_entity` bug.

## Capabilities

### New Capabilities

- `bot-api-testing`: Capability covering the HTTP-driven test harness
  for the `botCtl` Flask service. Owns requirements for how scenarios
  drive a running instance, how session/sequence/context state is
  propagated, and how response fields are surfaced for assertions.

### Modified Capabilities

- *(none)* — no existing capability has its requirements changed by
  this change.

## Impact

- New file: `botApiTest.py` at the repository root.
- Runtime dependency: the standard library plus `requests` (already
  used by `botActions.py` / `botLlm.py`).
- Requires a `botCtl.py` instance reachable at the configured URL
  (default `http://127.0.0.1:11354`) — the script is a client, not a
  test runner that boots the service itself.
- No changes to `botCtl.py`, `botActions.py`, `botIntents.py`,
  `botDecoder.py`, or any data/vector files.
- No new packages required; `requests` is already imported elsewhere
  in this project.
