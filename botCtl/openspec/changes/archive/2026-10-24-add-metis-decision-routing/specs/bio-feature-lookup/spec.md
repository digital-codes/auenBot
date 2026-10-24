# Spec Delta

## MODIFIED Requirements

### Requirement: bio_feature returns features for the resolved entity type

With Metis routing enabled, the feature for a bio answer SHALL NOT be the
decoded intent's suffix. Stage 2 decides the category
(verhalten/eigenschaften/beschreibung); the answer merges all filled columns
of that category's column group and attaches media from the entity's Links
(audio for verhalten, otherwise an image). When the category decision is
undecided (top-2 probability margin below 0.10 or confidence below 0.35),
the answer merges all filled columns of all categories and offers 2-3
feature options drawn from the entity's actually filled columns. The legacy
per-intent feature column flow remains in force when Metis routing is
disabled. The `bio_feature` action SHALL still restrict feature retrieval to
entities whose `Typ` matches the type implied by the request (`Tier` for
animals, `Pflanze` for plants) and MUST NOT return features from entities of
a different type.

#### Scenario: Above-threshold plant matches are ignored for an animal query

- **WHEN** the user query resolves primarily to an animal entity
  and other above-threshold matches are plants
- **THEN** the returned features are restricted to the animal
  match and contain no plant data

#### Scenario: Food question answers from the behaviour group

- **WHEN** the question is "Was frisst der Eisvogel?" and stage 2 chooses
  verhalten
- **THEN** the answer contains Nahrung and the other filled verhalten
  columns (e.g. Lebensraum, Fortpflanzung) with an audio link when present

#### Scenario: Undecided category offers feature options

- **WHEN** stage 2 returns a confidence below 0.35 for a vague question
  about the Gelbbauchunke
- **THEN** the answer merges filled columns from all categories and offers
  2-3 of the entity's filled features as options
