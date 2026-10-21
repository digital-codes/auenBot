# Spec Delta

## Purpose

Owns the contract for the `bio_feature` action: how a user query is
resolved to a single entity of the right type, how the requested
feature key is mapped to data fields, and how the response is
shaped so callers receive only the features of the resolved entity
— never a mix of features from unrelated entities.

## ADDED Requirements

### Requirement: bio_feature resolves to a single best-matching entity

The `bio_feature` action SHALL resolve the user query to a single
best-matching entity of the requested type (e.g. `Tier` or
`Pflanze`) before retrieving features. If multiple entities score
above the configured fuzzy-match threshold, the action SHALL pick
the entity with the highest score and MUST NOT silently return
features for any other above-threshold entity.

#### Scenario: Clear best match returns only that entity's features

- **WHEN** the user query has a single best-matching entity (e.g.
  *"was macht das Reh"* resolves to `Reh` with 100% match score)
- **THEN** the returned `text`, `image`, and `audio` fields contain
  features for `Reh` only, and MUST NOT contain feature fields from
  any other entity (e.g. `Regenbremse`, `Köcherfliegen`)

#### Scenario: Multiple above-threshold matches still pick the best

- **WHEN** the user query has several above-threshold matches
  (e.g. `Reh` 100% and `Regenbremse` 72%) but a clear best score
- **THEN** only the best-scoring entity's features are returned,
  and the lower-scoring matches are discarded

### Requirement: bio_feature returns features for the resolved entity type

The `bio_feature` action SHALL restrict feature retrieval to
entities whose `Typ` matches the type implied by the request
(`Tier` for animals, `Pflanze` for plants). It MUST NOT return
features from entities of a different type, even if those entities
score above the fuzzy-match threshold.

#### Scenario: Above-threshold plant matches are ignored for an animal query

- **WHEN** the user query resolves primarily to an animal entity
  and other above-threshold matches are plants
- **THEN** the returned features are restricted to the animal
  match and contain no plant data

### Requirement: bio_feature response shape is text, image, and audio

The `bio_feature` action SHALL return a result with the keys
`text` (list of strings), `image` (list of image URLs), and
`audio` (list of audio URLs). When the requested feature key is
`definition`, the action SHALL attempt to populate all three lists
from the resolved entity's data.

#### Scenario: definition key returns text and media

- **WHEN** `bio_feature` is invoked with `feature: "definition"`
  for an entity that has text, an image, and an audio entry
- **THEN** the result's `text` list is non-empty for that entity,
  `image` contains the entity's first image URL, and `audio`
  contains the entity's first audio URL

#### Scenario: No media available returns empty media lists

- **WHEN** `bio_feature` is invoked for an entity that has text
  but no image or audio entry
- **THEN** `text` is populated and `image` and `audio` are empty
  lists

### Requirement: bio_feature reports no-data gracefully

The `bio_feature` action SHALL return a non-empty `text` list
containing a localised *"no information available"* message when
the resolved entity has no value for any of the requested feature
keys, rather than returning empty lists.

#### Scenario: Entity exists but has no values for the requested keys

- **WHEN** the resolved entity exists but every requested feature
  key is `None` or missing
- **THEN** the result's `text` list contains a single message that
  tells the user no information is available for that entity and
  feature, and `image`/`audio` are empty lists

### Requirement: bio_feature debug output names the chosen entity

When the bot runs with debug output enabled, the `bio_feature`
action's debug log MUST include, in order: the resolved entity
name and type, the set of candidate matches that were considered
(scores and names), and the per-feature values that were emitted
for that single entity. Debug output MUST be readable enough that a
human can confirm from one log run that the response came from
exactly one entity.

#### Scenario: Debug log shows single-entity decision

- **WHEN** the bot runs with debug enabled and `bio_feature` is
  invoked
- **THEN** the debug log contains exactly one line naming the
  resolved entity, one line listing the candidate matches with
  scores, and per-key lines showing only that entity's values
