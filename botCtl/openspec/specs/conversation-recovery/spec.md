# conversation-recovery Specification

## Purpose

Lets the conversation recover from repeated misunderstanding and keep
going after a good answer: a configurable breakout resets the
conversation after repeated no-match turns, and a focus snapshot lets
natural follow-up questions reuse the entity of the previous answer.

## Requirements

### Requirement: Consecutive no-match turns are counted per conversation

The system SHALL count consecutive no-match outcomes in the
conversation context. A no-match outcome is a turn where the entity
requirement cannot be resolved (staged matching and arbiter failed) or
a reply fails to match offered options. A focus-filled follow-up turn
SHALL count as a miss when it cannot answer (the requested feature has
no data for the focused entity). A turn that answers with content —
from a direct match or from a focus-filled follow-up — SHALL reset the
count to zero.

#### Scenario: Misses accumulate across turns

- **WHEN** user inputs fail entity resolution in consecutive turns
  against an active intent (e.g. gibberish twice)
- **THEN** the context returned each turn carries the incremented miss
  count

#### Scenario: A reply that never matches options counts as a miss

- **WHEN** entity options were offered in a previous turn and the
  user's reply matches none of them
- **THEN** that turn increments the miss count like any other no-match
  outcome

#### Scenario: Unhelpful focus-filled turns count, helpful ones reset

- **WHEN** focus-filled follow-ups run against an entity whose feature
  has no data (each falling into the feature-fallback prompt) while
  interleaved replies never match the offered options
- **THEN** those turns increment the miss count until the breakout
  triggers, whereas a focus-filled follow-up that answers with content
  resets the count to zero

#### Scenario: Successful match resets the miss counter

- **WHEN** a no-match turn is followed by a turn whose entity matches
  successfully and the answer carries content
- **THEN** the stored miss count returns to zero, so a later rough
  patch gets the full configured number of retries again

### Requirement: Reaching the configured miss limit resets the conversation

When the miss count reaches a configurable limit (default 3), the
system SHALL clear the conversation context entirely (intent,
requirements, options, focus) and reply with a message that invites the
user to ask again. The session id and sequence numbering SHALL continue
unchanged.

#### Scenario: Breakout triggers at the configured count

- **WHEN** the miss count reaches the configured limit (e.g. the third
  consecutive miss with limit 3)
- **THEN** the reply invites the user to try again, and the context
  returned for the next turn is empty — no intent, no requirements, no
  focus

#### Scenario: Threshold is configurable

- **WHEN** the service is configured with a different miss limit (e.g.
  2 instead of the default 3)
- **THEN** the breakout triggers exactly at the configured count

#### Scenario: Session continuity is kept across a breakout

- **WHEN** the breakout reset triggers
- **THEN** the session id and sequence numbering continue unchanged, so
  history rows stay grouped under the same session

### Requirement: Completed answers set a focus snapshot

After a completion flow answers successfully with a resolved entity,
the system SHALL store that entity (with its type) as the conversation
focus in the context. Each subsequent successful completion SHALL
replace the focus. The group/options path SHALL NOT set a focus, since
nothing was answered yet.

#### Scenario: Focus is set after a completed answer

- **WHEN** the completion flow answers about *Reh*
- **THEN** the returned context carries a focus referencing the Reh
  entity and its type

#### Scenario: Focus follows the most recent answer

- **WHEN** two completions ran in sequence (first about *Reh*, then
  about *Moorfrosch*)
- **THEN** the focus references *Moorfrosch*, the most recently
  completed entity

### Requirement: Entity-less feature follow-ups resolve against the focus

When a later turn decodes to a slot-filling intent of the
tp/tiere/pflanzen families, the entity requirement cannot be resolved
by staged matching or the arbiter, and a focus exists, the system SHALL
fill the entity requirement from the focus and continue the pipeline —
the feature comes from the newly decoded intent. When the input already
names a resolvable entity, the focus SHALL NOT be consulted.

#### Scenario: Feature follow-up reuses the previous entity

- **WHEN** the previous turn answered about *Reh* and the next input is
  a feature question without an entity (e.g. *und wie groß ist sie?*)
  that decodes to a size intent
- **THEN** the entity requirement is filled from the focus (*Reh*) and
  the reply addresses that entity — with the requested feature's
  content, or the feature-fallback options when the entity has no data
  for it (see the feature-fallback requirement)

#### Scenario: Focus does not fire when the entity resolves normally

- **WHEN** the input already names a resolvable entity (e.g.
  *moorfrosch*)
- **THEN** the normal matching path resolves the entity and the focus
  is not consulted

#### Scenario: Breakout clears the focus

- **WHEN** the breakout reset triggers
- **THEN** the focus is removed together with the rest of the context,
  so follow-ups cannot resolve against a stale entity after a restart

### Requirement: Empty feature lookups offer available features

When a resolved entity has no data for the requested feature, the
system SHALL inform the user that the feature is unavailable and offer
up to 3 other features of that entity that have data, as selectable
options. Selecting an offered feature SHALL answer with that feature's
content. If the entity has no other features with data, the existing
no-information reply SHALL be used.

#### Scenario: Size follow-up on a feature without data offers alternatives

- **WHEN** the previous turn answered about *Reh* and the next input is
  a size question without an entity (e.g. *und wie groß ist sie?*) but
  the entity has no data for that feature (Reh has no `Größe` field)
- **THEN** the reply informs the user that the feature is unavailable
  and offers up to 3 other features of the entity (with data) as
  selectable options, and the context keeps the entity

#### Scenario: Picking an offered feature answers with its content

- **WHEN** feature-fallback options were offered and the user picks one
  (e.g. *Fortpflanzung*)
- **THEN** the reply contains that feature's real content for the
  entity

#### Scenario: Entity without any other data keeps the existing reply

- **WHEN** the resolved entity has no data for the requested feature
  and no other data-carrying fields either
- **THEN** the existing no-information reply is used unchanged

### Requirement: Completed answers record the conversation trace

On a successful completion the system SHALL store the answered input
and the completed intent in the context as `last_input` and
`last_intent` so the carried context reflects the conversation trace.

#### Scenario: Completion writes the trace fields

- **WHEN** a completion flow finishes successfully
- **THEN** the returned context carries `last_input` with the user
  input of that turn and `last_intent` with the completed intent name
