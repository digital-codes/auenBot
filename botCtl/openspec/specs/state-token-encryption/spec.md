# state-token-encryption Specification

## Purpose

Makes the conversation state that the client carries between requests an
opaque, tamper-evident encrypted token, so the backend can stay stateless
without trusting or exposing plaintext context on the wire.

## Requirements

### Requirement: Conversation state is carried as an opaque sealed token

The system SHALL encode the conversation context as a single opaque
state token in API responses, and SHALL accept the conversation state
only in the form of such a token in subsequent requests. The token
MUST be opaque: its content SHALL NOT be readable or modifiable by the
client. The token MUST be bound to the session it was issued for.

#### Scenario: New session starts without a token

- **WHEN** a request with no session id and no state token arrives
- **THEN** the backend starts a fresh conversation and the response
  carries a newly sealed state token

#### Scenario: Continuing session echoes the token

- **WHEN** a request carries the state token from a previous response
  of the same session
- **THEN** the backend restores the conversation context from it,
  processes the message, and the response carries a freshly sealed
  state token reflecting the updated context

#### Scenario: Token content is opaque to the client

- **WHEN** a client inspects a state token from any response
- **THEN** no context field (intent, entity, feature, options) can be
  read from the token without the server secret

### Requirement: Tampered or malformed tokens are rejected

The system SHALL reject a request whose state token fails integrity
verification (any bit flipped, payload modified, or version prefix
unknown) or cannot be parsed, with HTTP 400, and MUST NOT process the
message with any partially restored context.

#### Scenario: Modified token is rejected

- **WHEN** a request carries a state token whose ciphertext was altered
  after it was issued
- **THEN** the backend responds with HTTP 400 and no intent pipeline
  processing occurs

#### Scenario: Malformed token is rejected

- **WHEN** a request carries a string that is not a valid state token
  (wrong version prefix, invalid base64, wrong payload shape)
- **THEN** the backend responds with HTTP 400

### Requirement: Tokens are bound to their session

The system SHALL refuse to open a state token under a session id other
than the one it was sealed for, with HTTP 400.

#### Scenario: Token replayed in another session is rejected

- **WHEN** a state token issued for session A is sent in a request for
  session B
- **THEN** the backend responds with HTTP 400 and does not start a
  conversation with session A's context in session B

### Requirement: Sealing key derives from a server secret and the session id

The system SHALL derive the token encryption key from a server secret
combined with the session id, such that knowledge of the session id
alone is insufficient to forge tokens. The server secret SHALL be taken
from the `STATE_TOKEN_SECRET` environment variable. When the variable
is unset, the system SHALL generate a random secret at startup, print
a warning that tokens will not survive a restart, and continue
operating.

#### Scenario: Configured secret survives a restart

- **WHEN** `STATE_TOKEN_SECRET` is set to the same value across a
  backend restart and a client presents a token issued before the
  restart
- **THEN** the backend opens the token and the conversation continues

#### Scenario: Random secret invalidates tokens across a restart

- **WHEN** `STATE_TOKEN_SECRET` is not set, the backend restarts, and a
  client presents a token issued before the restart
- **THEN** the backend rejects the token with HTTP 400, and a warning
  was printed at startup

### Requirement: User-facing option lists accompany the sealed state

When a response's context carries an option list (intent choices,
species group matches, or feature-fallback alternatives), the response
SHALL additionally include that list under an `options` field in
plaintext, next to the sealed state token. The list SHALL have exactly
the shape of the context's option entries. Responses without options
SHALL omit the field.

#### Scenario: Option prompt carries the choices in plaintext

- **WHEN** the backend responds with an option prompt (e.g. the five
  frosch species as a group match)
- **THEN** the response contains the sealed `state` token AND an
  `options` field listing exactly those choices with their titles

#### Scenario: Responses without options omit the field

- **WHEN** the backend responds to a turn that resolved without any
  option flow (e.g. a direct species answer)
- **THEN** the response contains the sealed `state` token and no
  `options` field

### Requirement: Exposing options does not weaken the seal

The options SHALL remain part of the sealed state as well, and no other
context field (intent, entity, feature, focus) SHALL be exposed outside
the token.

#### Scenario: Sealed state remains opaque for protected fields

- **WHEN** a client inspects a response that carries plaintext options
- **THEN** intent, entity, feature, and focus remain readable only
  inside the sealed token, and the options inside the token match the
  plaintext list
