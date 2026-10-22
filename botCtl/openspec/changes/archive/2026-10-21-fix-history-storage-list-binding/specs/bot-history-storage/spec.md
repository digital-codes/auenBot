# Spec Delta

## Purpose

Owns the contract for persisting every conversation step (RX and
TX) of the bot to the `history` table in `bot_history.db`: which
columns a row carries, and the guarantee that any message payload
the conversation pipeline can produce is storable — a row MUST NOT
be silently dropped because a field arrived as a list or dict.

## ADDED Requirements

### Requirement: Every processed conversation step is persisted

The system SHALL insert one `history` row for each received (RX)
message and one for each transmitted (TX) message of a processed
request. A row SHALL carry at minimum: receive time, the full
message as JSON, message type (`RX` or `TX`), status, session id,
sequence number, language, and the message text.

#### Scenario: RX row is written for a valid request

- **WHEN** a valid request arrives at the API endpoint
- **THEN** a `history` row with `message_type = "RX"`, the
  request's session id, and its sequence number is persisted before
  the response is produced

#### Scenario: TX row is written for the bot's answer

- **WHEN** the pipeline produces a bot answer (text, option
  prompt, or error text) for a request
- **THEN** a `history` row with `message_type = "TX"`, the same
  session id, and the incremented sequence number is persisted

### Requirement: List- and dict-shaped field values do not break persistence

The system SHALL coerce every value bound to a `history` column to
a type SQLite supports (`str`, `int`, `float`, `bool`, `None`,
`datetime`). Specifically, when the outgoing message's text is a
list of strings, the persisted `message_text` SHALL be a single
string containing all list entries, and MUST NOT be a list.
List- or dict-shaped values in any other column SHALL be serialized
to their JSON representation instead of causing a binding error.

#### Scenario: List-shaped answer text is stored as one string

- **WHEN** an intent action returns an answer whose `text` is a
  list of strings (e.g. a bio-feature answer with several feature
  values)
- **THEN** the persisted TX row has a `message_text` of type
  `str` containing every entry of that list, and no
  `sqlite3.ProgrammingError` occurs

#### Scenario: Sentinel no-data text is stored

- **WHEN** an intent action returns the *"no information
  available"* answer, whose `text` is a one-element list
- **THEN** the persisted TX row's `message_text` is a `str` equal
  to that message, and no binding error occurs

### Requirement: Storage failure never fails the conversation

If persisting a `history` row fails, the system SHALL still return
the conversation response to the caller. The failure SHALL be
logged (visible when debug output is enabled) and MUST NOT be
raised to the API response path.

#### Scenario: Persistence error is logged but response still returns

- **WHEN** a history insert fails for any reason
- **THEN** the API still returns the conversation response with
  its normal status code, and the failure is reported in the
  service log
