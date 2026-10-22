# Spec Delta

## ADDED Requirements

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
