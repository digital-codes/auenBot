# Spec Delta

## MODIFIED Requirements

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
