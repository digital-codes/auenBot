# entity-resolution Specification

## Purpose

Owns the mapping of German user input to entity items for intent
requirements: token-level matching with German plural
normalization, exact-stem matching before fuzzy fallback, and group
handling that offers the matching species as options when one query
stem spans several entities.

## Requirements

### Requirement: German plural forms match their singular entities

Entity matching SHALL compare a German-morphology stem of the user
input against stems of the entity item names: lowercase, umlaut
stem mapping (ä→a, ö→o, ü→u) and German plural suffix stripping
(-e, -en, -n, -er, -s, -ern). A plural user term (e.g. *frösche*)
SHALL match the entity names sharing its singular stem (e.g.
*Teichfrosch*, *Moorfrosch*), which a whole-sentence fuzzy match
MUST be required to achieve.

#### Scenario: Plural query reaches frog species

- **WHEN** the user input contains the plural term *frösche* and
  the entity items include names containing the stem *frosch*
- **THEN** the entity match is resolved within the *frosch*-stem
  items, and MUST NOT resolve to an entity whose name does not
  contain that stem (e.g. *Windröschen*)

#### Scenario: Singular query still matches

- **WHEN** the user input contains the singular term *frosch* or a
  full species name such as *Moorfrosch*
- **THEN** the entity match resolves within the *frosch*-stem items
  with the same precedence rules as plural input

### Requirement: Token-level matching precedes fuzzy fallback

Entity matching SHALL first attempt exact stem containment between
input tokens and entity name stems, considering the full entity
name and its individual words. Fuzzy scoring SHALL be used only as
a fallback for input that has no exact stem containment, and a
whole-sentence fuzzy score alone SHALL NOT select an entity when no
token-level stem relationship exists.

#### Scenario: Unrelated sentence does not select a lookalike entity

- **WHEN** the user input shares only a loose substring with an
  entity name (e.g. *frösche* vs. *Windröschen*) and has no stem
  containment with any entity name other than the *frosch* group
- **THEN** the match result is derived from the stem relationship,
  and the lookalike entity MUST NOT be selected

#### Scenario: No stem relationship yields no match

- **WHEN** the user input has no stem containment with any entity
  item and no fuzzy fallback score reaches the configured threshold
- **THEN** the matcher reports no match instead of returning an
  arbitrary low-confidence entity

### Requirement: A group match offers the species as options

When the input stem matches more than one entity item, the
completion flow SHALL present all matching entity items as
selectable options in the context (the existing options mechanism)
with a prompt asking the user to choose, and MUST NOT fill the
entity requirement with an arbitrary group member. When exactly one
entity item matches, the requirement SHALL be filled directly as
today.

#### Scenario: Plural frog query offers five species

- **WHEN** the user input *gibt es da frösche* is processed and the
  entity items contain five *frosch*-stem species
- **THEN** the response asks the user to choose, the context
  carries options for exactly those five species, the entity
  requirement is not yet filled, and no species is answered
  directly

#### Scenario: Single-stem match fills the requirement directly

- **WHEN** the user input names a species unambiguously (e.g.
  *Moorfrosch*) so exactly one entity item matches
- **THEN** the entity requirement is filled with that species and
  the flow completes without an options prompt

### Requirement: A selection from offered options fills the requirement

When entity species options were offered in the previous turn, the
user's next reply SHALL be matched against the offered options by
the existing option-matching flow. A matched reply SHALL fill the
entity requirement with the chosen entity, and the flow SHALL
continue as for a direct single match.

#### Scenario: User picks one of the offered species

- **WHEN** species options were offered in the previous turn and
  the user replies with one of them (e.g. *Teichfrosch*)
- **THEN** the existing option-matching flow resolves the reply to
  that species, fills the entity requirement with it, and the flow
  continues as for a direct single match
