# Spec Delta

## ADDED Requirements

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
