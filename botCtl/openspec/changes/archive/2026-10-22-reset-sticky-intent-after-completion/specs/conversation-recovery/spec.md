# Spec Delta

## MODIFIED Requirements

### Requirement: Consecutive no-match turns are counted per conversation

The system SHALL count consecutive no-match outcomes in the
conversation context. A no-match outcome is a turn that ends without an
answer: the entity requirement cannot be resolved (staged matching and
arbiter failed), a reply fails to match offered options and fallback
re-decoding finds no other intent, or a focus-filled follow-up cannot
answer (the requested feature has no data for the focused entity). A
turn that answers with content — from a direct match, from a
focus-filled follow-up, or from an intent that took over after fallback
re-decoding — SHALL reset the count to zero.

#### Scenario: Misses accumulate across turns

- **WHEN** user inputs fail entity resolution in consecutive turns
  against an active intent (e.g. gibberish twice)
- **THEN** the context returned each turn carries the incremented miss
  count

#### Scenario: A reply that never matches options counts as a miss

- **WHEN** entity options were offered in a previous turn and the
  user's reply matches none of them and no other intent can be decoded
- **THEN** that turn increments the miss count like any other no-match
  outcome

#### Scenario: A turn rescued by re-decoding is not a miss

- **WHEN** options are pending and the user's reply matches none of
  them, but the input decodes to another intent that answers
- **THEN** the turn does not increment the miss count (the count resets
  as for any answered turn), and the pending flow is abandoned

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
