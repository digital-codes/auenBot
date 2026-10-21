# Design

## Context

`botCtl.py` exposes a single Flask endpoint `POST /api` (default
`http://127.0.0.1:11354/api`) that consumes a JSON payload matching
`botSchema.json` and returns a JSON response containing the new
`context`, the bot's `message`, and the running `session` /
`sequence`. The server persists `session`/`sequence`/`context` in
SQLite (`bot_history.db`) and uses the in-memory
`BotIntent`/`BotDecoder`/`BotAction` state machine to resolve the
user's intent, fill required context slots, and dispatch the action.

We need a thin external client script that drives that HTTP API the
same way the real front-end would, so we can reproduce the *Reh*
behaviour seen in `botCtl.log` (intent `tp_definition`, entity
`Reh`, but bio-feature output that mixes unrelated entities). See
`proposal.md` for motivation. The specs in
`specs/bot-api-testing/spec.md` define the observable behaviour the
script must satisfy.

The project already depends on `requests` (used by `botActions.py`
and `botLlm.py`), so the script can reuse it without adding new
third-party packages. There is no existing test harness in the
repository, so this is a new pattern.

## Goals / Non-Goals

**Goals:**

- Provide a runnable Python script at the repository root that
  drives the existing `botCtl` HTTP endpoint as an external client.
- Keep the script small, dependency-light, and easy to extend with
  new scenarios in subsequent change requests.
- Reproduce the *Reh* scenario from `botCtl.log` so the suspected
  bio-feature bug can be observed by a human or CI runner.

**Non-Goals:**

- Fixing the suspected `BotAction.find_entity` bug. That belongs in
  a separate change request that can leverage the script added here
  to verify the fix.
- Asserting on the bio-feature text in this change. The script only
  prints the text so the bug becomes visible; assertions are added
  in a later change once we agree on the expected behaviour.
- Booting `botCtl` from inside the script. The script is a pure
  client and expects an externally running service (matches the
  `botCtl.py` `app.run(host="0.0.0.0", port=11354)` default).
- Replacing the manual reproduction steps shown in `botCtl.log`.

## Decisions

### Decision 1: Plain script with a tiny per-scenario state object

The script exposes a small `BotApiClient` (a thin wrapper over
`requests.Session`) plus a `ScenarioState` dataclass that tracks
`session`, `sequence`, `context`, and the last `message`. Each
scenario is a function that takes a `BotApiClient` and a
`ScenarioState`, calls `client.send(scenario_state, text)`, and
inspects the response.

- **Why**: scenarios are easier to add (drop in a function) and the
  state-machine plumbing is in one place, satisfying the "script is
  safe to extend with more scenarios" spec requirement.
- **Alternative considered**: a `unittest`/`pytest` test class.
  Rejected because the user asked for a runnable script, not a test
  framework, and because we are not yet asserting on outputs in this
  change — assertions belong in a later change once the bug is
  characterised.

### Decision 2: Reuse `requests` and the existing schema, do not
duplicate it

`requests` is already used in this project. The script imports
`requests` directly and uses the `botSchema.json` field names as
literals — it does **not** attempt to validate its own payloads
against the schema.

- **Why**: keeps the script dependency-free of the server's runtime
  imports; avoids touching `botCtl.py` from this change.
- **Alternative considered**: importing `botSchema.json` and
  validating before sending. Rejected as overkill for a manual
  repro script — the server already validates payloads and returns
  400 on schema errors, which is observable from the response.

### Decision 3: Default endpoint `http://127.0.0.1:11354/api`,
overridable via CLI flag

The default URL matches the hard-coded `app.run(host="0.0.0.0",
port=11354, ...)` in `botCtl.py`. The script reads a single
`--url` argument so maintainers can point it at staging or a
containerised instance.

- **Why**: zero-config for the documented local-run case; one CLI
  flag is enough for non-default setups.
- **Alternative considered**: environment variable
  (`BOTCTL_URL`). Rejected because the project does not already use
  env vars for configuration and a CLI flag is more discoverable
  for a one-off script.

### Decision 4: Print full response fields for the *Reh* scenario;
no assertions

The *Reh* scenario prints `intent`, `entity`, `feature`, the full
`message.text`, and any `error` returned by the server. It does not
assert on equality with expected text.

- **Why**: the suspected bug is precisely that the bio-feature text
  is wrong; asserting on the wrong text today would make the script
  red immediately. Printing the full response lets a human see the
  bug without locking in a (currently broken) expected value.
  Assertions are deferred to the follow-up change that fixes the
  underlying bug.
- **Alternative considered**: assert that `message.text` mentions
  *Reh*. Rejected because a premature assertion would either pass
  against broken output or force us to encode the wrong text into
  the test suite.

## Risks / Trade-offs

- [Risk] The script depends on a `botCtl` process being already
  running → **Mitigation**: at start-up, the script does a `GET
  /api` health check (the server already exposes this as
  `health_check`) and prints a clear error if the server is
  unreachable, so the failure mode is obvious.
- [Risk] The script's printed output becomes noisy for long feature
  blobs (the *Reh* bio-feature dump in `botCtl.log` already exceeds
  2000 chars) → **Mitigation**: print full `message.text`
  unconditionally for the *Reh* scenario (we want to see the bug),
  but truncate other fields like `context` to a configurable length
  to keep the output readable. Follow-up scenarios can override the
  truncation policy.
- [Risk] Mixing the new test pattern with the existing
  `botCtl.py` debug logging (which is very chatty) could obscure the
  scenario output → **Mitigation**: the script does not toggle the
  server's `DEBUG` flag; it only consumes the HTTP response. The
  server's `botCtl.log` remains the canonical place to inspect
  server-side logs.
- [Trade-off] Choosing `requests` over `httpx`/`urllib` ties the
  script to the same HTTP library the rest of the project uses,
  keeping the dependency footprint consistent.

## Migration Plan

- No deployment step: the script is a new file in the repo and has
  no effect on the running service.
- Rollback: deleting `botApiTest.py` reverts this change entirely;
  no other files are modified.

## Open Questions

- Should the *Reh* scenario also persist its request/response pair
  to a JSON fixture file for later comparison? Deferred — useful for
  the eventual fix-verification change, not needed for the
  reproduction step.
