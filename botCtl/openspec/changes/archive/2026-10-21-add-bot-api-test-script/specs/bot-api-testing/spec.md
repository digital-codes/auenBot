# Spec Delta

## Purpose

Lets maintainers drive a running `botCtl` Flask service from an
external script, so conversation scenarios can be exercised and
captured without booting the service in-process. The first scenario
reproduces the *Reh* sample from `botCtl.log` to make the suspected
bio-feature lookup bug observable.

## ADDED Requirements

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
(`input`, `session`, `sequence`, `lang`, `message`, `context`) matches
the JSON Schema in `botSchema.json`. The `message.text` field MUST
carry the user input the scenario wants to send.

#### Scenario: First request starts a new session
- **WHEN** the *Reh* scenario sends its first request
- **THEN** the request contains no `session` value and `sequence` 0,
  and the bot assigns and returns a fresh `session` id

#### Scenario: Subsequent requests reuse the returned session
- **WHEN** the script receives a response that contains a `session`
  id and a `sequence` number
- **THEN** the next request in the same scenario carries that
  `session` id and the updated `sequence`

### Requirement: Response state is propagated between calls

The script SHALL feed the `context` from each response into the next
request in the same scenario, so the bot's state machine is exercised
end-to-end (intent → entity → feature lookup) instead of being reset
between calls.

#### Scenario: Context carries resolved entity forward
- **WHEN** a scenario sends `"was macht das reh"` and the response
  `context` contains `entity: "Reh"`
- **THEN** the next request in the same scenario carries
  `context.entity: "Reh"` so the bio-feature lookup runs with that
  entity preserved

### Requirement: Scenario for the Reh log sample

The script SHALL include a first scenario that posts
`"was macht das reh"` in German (`lang: "de"`) against a fresh
session, captures the response, and prints the resolved intent, the
resolved entity, the resolved feature, and the returned
`message.text` for human inspection.

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
