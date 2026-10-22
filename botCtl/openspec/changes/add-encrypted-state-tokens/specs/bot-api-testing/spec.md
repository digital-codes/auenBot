# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

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
