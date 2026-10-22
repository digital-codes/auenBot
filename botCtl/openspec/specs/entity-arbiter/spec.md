# entity-arbiter Specification

## Purpose

Recovers entity resolution when the staged matcher finds nothing: an
LLM arbiter classifies the user's term against the domain candidate
lists and, only after strict validation, fills the entity requirement
so the pipeline can continue.

## Requirements

### Requirement: Failed entity matching triggers the arbiter

When the completion flow cannot resolve the `entity` requirement
because the staged matcher reports no match, the system SHALL make
exactly one LLM chat call that classifies the user input against the
entity name lists of the three domain files (`entity_tiere.json`,
`entity_pflanzen.json`, `entity_aue.json`) and identifies the entity
name the input refers to, including when it appears inside a longer
sentence. Successful match paths MUST NOT trigger the arbiter.

#### Scenario: Auen name not in the intent's pool is recovered

- **WHEN** the user asks about *hartholzaue* and the intent's entity
  pool does not contain that name, so the staged matcher reports no
  match
- **THEN** the arbiter classifies the term into the Auen domain and
  returns *Hartholzaue*, the entity requirement is filled with it, and
  the pipeline continues to answer with real entity content

#### Scenario: In-sentence term is recovered

- **WHEN** the user input mentions a valid entity name inside a longer
  sentence (e.g. *was ist die feuchtwiese im nsg*) and staged matching
  found nothing
- **THEN** the arbiter returns the matching entity name from the
  candidate lists and the requirement is filled with it

#### Scenario: Successful matches skip the arbiter

- **WHEN** the staged matcher resolves the entity requirement directly
  (single or group match)
- **THEN** no LLM call is made for entity resolution

### Requirement: Arbiter answers are validated strictly

The system SHALL accept an arbiter answer only when the returned name
matches exactly one entry of the claimed domain's candidate file
(case-insensitive, whitespace-trimmed) and the claimed domain matches
that file. Malformed output, unknown domains, names absent from the
claimed list, and `found: false` answers MUST be treated as "not
found". The entity requirement MUST NEVER be filled with a name that
is not present in one of the three candidate files.

#### Scenario: Hallucinated name is discarded

- **WHEN** the arbiter returns a name that is not an entry of the
  claimed domain's candidate file
- **THEN** the answer is discarded and the flow continues on the
  no-match path

#### Scenario: Cross-domain confusion is rejected

- **WHEN** the arbiter returns a name that exists in another domain's
  file than the claimed one
- **THEN** the answer is discarded and the flow continues on the
  no-match path

#### Scenario: Out-of-vocabulary term yields no match

- **WHEN** the user input names something absent from all three
  candidate files (e.g. *luchs*) and staged matching found nothing
- **THEN** the arbiter reports "not found" and the existing no-match
  behavior (error response / intent options) is used

### Requirement: Arbiter failure degrades to the existing no-match path

If the LLM call fails (network error, timeout, unparseable response),
the system SHALL continue exactly as if the arbiter had reported "not
found". The conversation response MUST NOT fail because of the
arbiter, and no additional error surface is introduced.

#### Scenario: LLM unavailable keeps the conversation working

- **WHEN** the arbiter's LLM call raises an exception or times out
  during entity recovery
- **THEN** the flow continues on the existing no-match path and the
  user receives the normal error response

### Requirement: Arbiter usage is observable

When the arbiter has run for a turn, the `llm_used` flag SHALL be
propagated so the stored history row reflects the LLM usage, matching
the existing intent tie-breaker's reporting.

#### Scenario: Arbiter turn is marked in history

- **WHEN** a turn required the arbiter to resolve the entity
  requirement (successfully or not)
- **THEN** the stored history rows for that turn carry
  `llm_used = true`
