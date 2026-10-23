# Spec Delta

## MODIFIED Requirements

### Requirement: Unmatched-input replies are user-appropriate

A reply that reports unmatched input SHALL be phrased for the user.
With Metis routing enabled, the transition decision between re-decode and
follow-up (whether an input continues the previous topic or starts a new
one) SHALL come from the stage-1 followup/new_topic answers instead of the
chat-LLM follow-up gate. The pending-completion capture rules are unchanged;
the LLM follow-up gate is only used when Metis routing is disabled.

#### Scenario: No-match reply text

- **WHEN** a turn ends in an unmatched-input error
- **THEN** the outgoing message text is a human-readable sentence in
  the user's language, not an internal identifier

#### Scenario: Follow-up after a completed bio answer

- **WHEN** a bio question was answered about the Eisvogel and the next
  input is "und was frisst sie?"
- **THEN** stage-1 followup is above 0.7, the focused entity is reused
  without an entity search, and the bio feature flow answers from the
  focus entity

#### Scenario: Topic switch after a completed bio answer

- **WHEN** the focus is the Eisvogel and the input is
  "Wo lebt der Moorfrosch?" with new_topic above 0.7
- **THEN** the focus is cleared and the entity search resolves the
  Moorfrosch
