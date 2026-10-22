# Spec Delta

## MODIFIED Requirements

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
