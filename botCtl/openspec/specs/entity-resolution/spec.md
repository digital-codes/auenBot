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
token-level stem relationship exists. A fuzzy fallback candidate
SHALL only be accepted when the input token's stem and the candidate
name's stem share their first three characters (umlaut-mapped on both
sides); a high fuzzy score alone MUST NOT select an entity whose name
stem starts differently.

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

#### Scenario: Lookalike with a different word start is rejected

- **WHEN** the user input token has a high fuzzy score against an
  entity name but their stems start differently (e.g. *feuchtwies*
  vs. *efeu* at 77.1, or *luchs* vs. *rotfuchs* at 80.0)
- **THEN** the fuzzy fallback does NOT select that entity and reports
  no match

#### Scenario: Genuine typo with a shared word start is accepted

- **WHEN** the user input token is a typo of an entity name sharing
  the stem prefix and exceeding the threshold (e.g. *feuchtwies* vs.
  *feuchtwiesen* at 90.9)
- **THEN** the fuzzy fallback selects that entity

### Requirement: A group match offers the species as options

When the input stem matches more than one entity item, the
completion flow SHALL present at most five matching entity items as
selectable options in the context (the existing options mechanism)
with a prompt asking the user to choose, and MUST NOT fill the
entity requirement with an arbitrary group member. When more than
five items match, the offered five SHALL be the best-ranked matches
in rank order if the match ranking distinguishes them; if all
matching items share the same rank (no ranking signal), five SHALL
be chosen at random. When exactly one entity item matches, the
requirement SHALL be filled directly as today. Groups of five or
fewer items SHALL be offered in full.

#### Scenario: Plural frog query offers five species

- **WHEN** the user input *gibt es da frösche* is processed and the
  entity items contain five *frosch*-stem species
- **THEN** the response asks the user to choose, the context
  carries options for exactly those five species, the entity
  requirement is not yet filled, and no species is answered
  directly

#### Scenario: Large group is capped at five options

- **WHEN** the input matches more than five entity items (e.g. an
  exact entity plus every loosely prefix-matching *Groß-* species)
- **THEN** the response offers at most five options — the
  best-ranked matches in rank order when ranks differ, otherwise a
  random five — and the entity requirement is not yet filled

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
