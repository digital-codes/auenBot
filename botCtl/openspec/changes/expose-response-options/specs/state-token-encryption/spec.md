# Spec Delta

## ADDED Requirements

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
