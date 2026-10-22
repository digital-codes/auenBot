# bot-api-testing Specification

## Purpose

Lets maintainers drive a running `botCtl` Flask service from an
external script, so conversation scenarios can be exercised and
captured without booting the service in-process. The first scenario
reproduces the *Reh* sample from `botCtl.log` to make the suspected
bio-feature lookup bug observable.

## Requirements

### Requirement: HTTP client drives a running botCtl instance

The test script SHALL send requests to the `POST /api` endpoint of a
`botCtl` Flask instance already running on the network. The default
endpoint MUST be `http://127.0.0.1:11354/api` and MUST be overridable
from the command line or a configuration value.

#### Scenario: Default endpoint reaches a locally running botCtl

- **WHEN** the script is invoked with no override and a `botCtl`
  instance is reachable at `http://127.0.0.1:11354/api`
- **THEN** the script sends a request to `http://127.0.0.1:11354/api`
  and receives a JSON response

#### Scenario: Override endpoint reaches a non-default host

- **WHEN** the script is invoked with a custom host/port override
- **THEN** the script sends the request to the overridden endpoint
  instead of the default

### Requirement: Request payload conforms to botSchema.json

The test script SHALL build request bodies whose top-level shape
(`input`, `session`, `sequence`, `lang`, `message`, `state`) matches
the JSON Schema in `botSchema.json`. The `message.text` field MUST
carry the user input the scenario wants to send. The `state` field
MUST carry the opaque state token from the previous response of the
same scenario; a scenario's first request MUST carry no `state` value.

#### Scenario: First request starts a new session

- **WHEN** the *Reh* scenario sends its first request
- **THEN** the request contains no `session` value, `sequence` 0, and
  no `state` token, and the bot assigns and returns a fresh `session`
  id and a sealed state token

#### Scenario: Subsequent requests reuse the returned session

- **WHEN** the script receives a response that contains a `session`
  id, a `sequence` number, and a state token
- **THEN** the next request in the same scenario carries that
  `session` id, the updated `sequence`, and that state token

### Requirement: Response state is propagated between calls

The script SHALL feed the state token from each response into the next
request in the same scenario, so the bot's state machine is exercised
end-to-end (intent → entity → feature lookup) instead of being reset
between calls. To keep existing context-level assertions working under
the opaque wire format, the script SHALL also make the decrypted
conversation context of the current turn available to scenarios, using
the same shared secret the service was started with.

#### Scenario: Context carries resolved entity forward

- **WHEN** a scenario sends `"was macht das reh"` and the decrypted
  response context contains `entity: "Reh"`
- **THEN** the next request in the same scenario carries the state
  token that encapsulates that context, so the bio-feature lookup runs
  with that entity preserved

#### Scenario: Context assertions read the decrypted token

- **WHEN** a scenario asserts on `intent`, `entity`, or `feature`
- **THEN** the assertion reads those values from the decrypted current
  response context, not from any plaintext field on the wire

### Requirement: Scenario for the Reh log sample

The script SHALL include a first scenario that posts
`"was macht das reh"` in German (`lang: "de"`) against a fresh
session, captures the response, and prints the resolved intent, the
resolved entity, the resolved feature, and the returned
`message.text` for human inspection. After the
`fix-bio-feature-lookup` change is applied, the script SHALL also
print a clear PASS/FAIL line that checks the returned
`message.text` against the expected post-fix behaviour: the text
MUST reference `Reh` content and MUST NOT reference the unrelated
species (`Köcherfliegen`, `Regenbremse`) that the pre-fix run
mixed in.

#### Scenario: Reh scenario prints the resolved intent

- **WHEN** the Reh scenario is executed against a running `botCtl`
- **THEN** the printed response context contains
  `intent: "tp_definition"` matching the `botCtl.log` run

#### Scenario: Reh scenario surfaces the bio-feature output

- **WHEN** the Reh scenario is executed against a running `botCtl`
- **THEN** the script prints the full `message.text` returned by the
  bot — including its `entity` and `feature` slots if the bot chose
  to preserve them in the response context — even when the text
  contains descriptions of unrelated entities (e.g.
  *Köcherfliegen*, *Regenbremse*), so the suspected bio-feature bug
  becomes visible. The script MUST NOT silently swallow the text or
  short-circuit on missing context slots.

#### Scenario: Reh scenario reports PASS after the bio-feature fix

- **WHEN** the Reh scenario is executed against a `botCtl` that
  already incorporates the `fix-bio-feature-lookup` fix
- **THEN** the script prints a `PASS` line on stdout when
  `message.text` references `Reh` content AND does NOT mention
  `Köcherfliegen` or `Regenbremse`, and prints a `FAIL` line (with
  the offending substring) when the response still mixes in those
  unrelated species

### Requirement: Script is safe to extend with more scenarios

The script SHALL structure scenarios as self-contained functions or
named blocks so additional scenarios (and later bug-specific
assertions) can be added without modifying the request/state
plumbing.

#### Scenario: Adding a new scenario does not require touching the
HTTP client

- **WHEN** a maintainer adds a new scenario that sends one user input
- **THEN** they only add a new function/block and call it from the
  script's entry point — the HTTP client, state propagation, and
  schema-building helpers remain unchanged

### Requirement: Entity-resolution regression samples

The script SHALL include entity-resolution regression samples that
exercise the Auen recovery and the hardened fuzzy fallback end-to-end
through the API: queries naming an Auen habitat entity resolve to real
entity content, and out-of-vocabulary terms do not resolve to any
species. All previously existing scenarios MUST keep passing
unchanged.

#### Scenario: Auen habitat query resolves to real content

- **WHEN** the samples send *hartholzaue*, *stillgewässer*,
  *feuchtwiesen*, or *magerrasen* against a fresh session
- **THEN** each query resolves to that habitat entity and the reply
  text carries the entity's real content rather than the
  *"Nichts gefunden"* error

#### Scenario: Out-of-vocabulary term does not resolve to a species

- **WHEN** the samples send a term absent from all entity files (e.g.
  *luchs*)
- **THEN** no species is answered and the flow stays on the no-match
  path (error response or intent options), never naming an unrelated
  species

#### Scenario: Existing samples stay green

- **WHEN** the full scenario suite runs after this change
- **THEN** the previously existing scenarios (reh, frosch, the 10
  entity samples, token security and restart scenarios, DB crosscheck)
  all report PASS unchanged

### Requirement: The script can start the service with a shared secret

The script SHALL be able to start the `botCtl` Flask service as a child
process with a `STATE_TOKEN_SECRET` value of its choosing, wait until
the service's health endpoint responds, run the requested scenarios,
and terminate the child process afterwards. Attaching to an
already-running instance MUST remain possible.

#### Scenario: Default run starts the service

- **WHEN** the script is invoked with no start-related options and no
  compatible service is reachable
- **THEN** the script starts a `botCtl` child process with a generated
  random `STATE_TOKEN_SECRET`, waits for the health endpoint, and runs
  the requested scenario against it

#### Scenario: Fixed secret for reproducible runs

- **WHEN** the script is invoked with an explicit secret value
- **THEN** the started service child uses exactly that
  `STATE_TOKEN_SECRET` value, and the script uses the same value to
  seal and open tokens

#### Scenario: Attach mode keeps working

- **WHEN** the script is invoked with the attach option against an
  already-running `botCtl` started outside the script
- **THEN** the script does not spawn a child process and uses the
  provided or environment-provided secret for token operations

### Requirement: Security and restart scenarios

The script SHALL include scenarios that verify the state token
contract end-to-end: a modified token is rejected with HTTP 400, a
token from another session is rejected with HTTP 400, a service
restart with the same secret continues an existing session from a
pre-restart token, and a service restart without a configured secret
rejects a pre-restart token.

#### Scenario: Tampered token is rejected

- **WHEN** the script flips a character inside the token part of a
  valid state token and sends it
- **THEN** the service responds with HTTP 400 and the script reports
  the rejection

#### Scenario: Foreign-session token is rejected

- **WHEN** the script sends a token sealed for session A inside a
  request for session B
- **THEN** the service responds with HTTP 400

#### Scenario: Same-secret restart resumes the session

- **WHEN** the script runs part of a scenario, restarts the child
  service with the same fixed secret, and continues with the last
  state token
- **THEN** the conversation continues from the pre-restart context

#### Scenario: Missing-secret restart invalidates tokens

- **WHEN** the script restarts the child service without
  `STATE_TOKEN_SECRET` set and sends a pre-restart token
- **THEN** the service responds with HTTP 400

### Requirement: Stored history matches the decrypted wire context

The script SHALL be able to verify that the service stores decrypted
plaintext in `bot_history.db`: after a completed turn, the most recent
rows for the session SHALL contain the same context content the script
derived from decrypting the response token.

#### Scenario: DB cross-check after a turn

- **WHEN** a scenario completes one request/response turn and the
  script reads the latest RX and TX rows for that session from
  `bot_history.db`
- **THEN** each row's stored context matches the corresponding
  decrypted wire context (request context for RX, response context for
  TX), and the row count for the session increases by two

### Requirement: Breakout and follow-up scenarios

The script SHALL include scenarios that exercise conversation recovery
end-to-end through the API: a configurable number of consecutive
no-match inputs triggers the breakout reset with an empty returned
context, and an entity-less feature follow-up after a completed answer
resolves against the previously answered entity. All previously
existing scenarios MUST keep passing unchanged.

#### Scenario: Breakout after the configured miss count

- **WHEN** the scenario sends consecutive unresolvable inputs (e.g.
  gibberish) against one session until the configured limit is reached
- **THEN** the final reply invites the user to try again, and the
  decrypted context of that response is empty (no intent, no
  requirements, no focus)

#### Scenario: Feature follow-up against the focus

- **WHEN** the scenario first sends *"was macht das reh"* (completed
  answer) and then an entity-less size question (e.g. *"und wie groß
  ist sie?"*)
- **THEN** the follow-up addresses the Reh entity: the reply either
  carries the requested feature's content or — since Reh has no
  `Größe` data — informs about the missing feature and offers up to 3
  available features as options; picking one of those options yields
  real content

#### Scenario: Existing scenarios stay green

- **WHEN** the full scenario suite runs after this change
- **THEN** the previously existing scenarios (reh, frosch, the 15
  entity samples, token security and restart scenarios, DB crosscheck)
  all report PASS unchanged

### Requirement: Option-prompt assertions cover the plaintext options field

The script SHALL verify that option-prompt responses carry the
plaintext `options` field alongside the sealed state token, with
exactly the choices the decrypted context contains. All previously
existing scenarios MUST keep passing unchanged.

#### Scenario: Frosch group response exposes the species options

- **WHEN** the frosch scenario sends *"gibt es da frösche"* and the
  response offers the five frosch species
- **THEN** the response carries an `options` field listing exactly
  those five species next to the sealed state token, and the decrypted
  context's options match it

#### Scenario: Option-less responses omit the field

- **WHEN** a scenario turn resolves without an option flow (e.g. the
  *reh* scenario)
- **THEN** the response carries no `options` field

#### Scenario: Existing scenarios stay green

- **WHEN** the full scenario suite runs after this change
- **THEN** the previously existing scenarios (reh, frosch, the 15
  entity samples, breakout, follow-up, token security and restart
  scenarios, DB crosscheck) all report PASS unchanged
