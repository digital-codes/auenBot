# Spec Delta

## ADDED Requirements

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
