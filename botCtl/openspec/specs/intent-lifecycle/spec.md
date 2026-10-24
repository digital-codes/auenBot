# intent-lifecycle Specification

## Purpose

Owns how a conversation turn is routed between a pending completion
flow and a freshly decoded intent: when a pending intent consumes the
input, when it stops doing so after it has answered, and what happens
when a pending flow cannot use the input.

## Requirements

### Requirement: A completed flow stops capturing later turns

The pending-intent marker in the conversation context SHALL be removed
once the completion flow has delivered its answer (all requirements
satisfied, action executed). From the next turn on, user input SHALL be
decoded as a new request; the answered intent SHALL NOT capture or
influence it. The durable memory fields (`last_intent`, `last_input`,
focus) SHALL be kept so focus-based follow-ups keep working.

#### Scenario: Measurement query answered, then an unrelated question

- **WHEN** a messdaten query was answered (e.g. the O3 value) and the
  next input is "wann fährt der bus"
- **THEN** the reply comes from the intent matching that question, not
  from the messdaten option menu

#### Scenario: Directly naming an intent after another completion

- **WHEN** a tp/tiere/pflanzen question was answered and the next input
  is "anreise"
- **THEN** the anreise flow starts (its own options or utterance), not
  the previously answered intent

#### Scenario: Durable memory survives the marker reset

- **WHEN** a completion answered about an entity and the next input is
  a focus follow-up (e.g. "und wie groß ist sie?")
- **THEN** the follow-up still resolves against the previous entity,
  because `last_intent`/`last_input`/focus remain in the context

### Requirement: A pending completion consumes replies matching its options

While a completion flow is pending (options offered, requirements
unfilled), a user reply that matches one of the offered options SHALL
be consumed by that flow and continue or complete it. This behavior is
unchanged.

#### Scenario: Option pick completes a pending flow

- **WHEN** the messdaten option menu is pending and the user replies
  "O3"
- **THEN** the flow answers with the current O3 value and no longer
  counts as pending afterwards

### Requirement: No-match against a pending completion falls back to re-decoding

When the user's input matches none of a pending completion's offered
options, the system SHALL re-decode the input before giving up. If
re-decoding finds another intent, that intent SHALL take over the turn
and answer with its own content, and the pending flow SHALL be
abandoned. Only when re-decoding also fails SHALL the no-match reply be
given and the turn be treated as a miss (see conversation-recovery).

#### Scenario: Unrelated question while options are pending

- **WHEN** the messdaten option menu is pending and the input is
  "gibt es was zu essen"
- **THEN** the reply comes from the verpflegung intent and the pending
  messdaten options are cleared

#### Scenario: Gibberish while options are pending stays a miss

- **WHEN** options are pending and the input matches no option and
  decodes to no intent (e.g. "mist")
- **THEN** the turn counts as a no-match miss per conversation-recovery

### Requirement: Unmatched-input replies are user-appropriate

A reply that reports unmatched input SHALL be phrased for the user.
Internal message keys (e.g. `no_match_completion`) and developer
placeholder strings (e.g. "Nichts gefunden... hier könnte ein
Vorschlag kommen") SHALL NOT be sent as chat text.
With SystemOne routing enabled, the transition decision between re-decode and
follow-up (whether an input continues the previous topic or starts a new
one) SHALL come from the stage-1 followup/new_topic answers instead of the
chat-LLM follow-up gate. The pending-completion capture rules are unchanged;
the LLM follow-up gate is only used when SystemOne routing is disabled.

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
