# Spec Delta

## ADDED Requirements

### Requirement: Post-completion turns reach newly decoded intents

The API test suite SHALL cover regression scenarios where an input that
belongs to its own intent is sent after a completion-style intent has
answered in the same session. Each such scenario SHALL assert that the
reply comes from the newly decoded intent (not from the previously
answered intent's options or error path) and that no internal error
keys appear in the reply text.

#### Scenario: Bus question after a completed measurement query

- **WHEN** a session first completes a messdaten query (menu → value
  answer) and then sends "wann fährt der bus"
- **THEN** the reply addresses the bus/anreise topic and the messdaten
  option menu is not re-presented

#### Scenario: Anreise question after a completed tp query

- **WHEN** a session first completes a tp/tiere/pflanzen question and
  then sends "anreise"
- **THEN** the reply starts the anreise flow and the pending-intent
  marker is gone from the sealed state

#### Scenario: Unrelated question while options are pending switches intent

- **WHEN** the messdaten option menu is pending and the session then
  sends "gibt es was zu essen"
- **THEN** the reply comes from the verpflegung intent and contains no
  internal error key
