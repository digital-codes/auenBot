# Spec Delta

## Purpose

Lets maintainers drive a running `botCtl` Flask service from an
external script, so conversation scenarios can be exercised and
captured without booting the service in-process. The first scenario
reproduces the *Reh* sample from `botCtl.log` to make the suspected
bio-feature lookup bug observable.

## MODIFIED Requirements

### Requirement: Scenario for the Reh log sample

The script SHALL include a first scenario that posts
`"was macht das reh"` in German (`lang: "de"`) against a fresh
session, captures the response, and prints the resolved intent, the
resolved entity, the resolved feature, and the returned
`message.text` for human inspection. After the
`fix-bio-feature-lookup` change is applied, the script SHALL also
print a clear PASS/FAIL line that checks the returned
`message.text` against the expected post-fix behaviour: the text
MUST reference `Reh` content and MUST NOT reference the unrelated
species (`Köcherfliegen`, `Regenbremse`) that the pre-fix run
mixed in.

#### Scenario: Reh scenario prints the resolved intent

- **WHEN** the Reh scenario is executed against a running `botCtl`
- **THEN** the printed response context contains
  `intent: "tp_definition"` matching the `botCtl.log` run

#### Scenario: Reh scenario surfaces the bio-feature output

- **WHEN** the Reh scenario is executed against a running `botCtl`
- **THEN** the script prints the full `message.text` returned by the
  bot — including its `entity` and `feature` slots if the bot chose
  to preserve them in the response context — even when the text
  contains descriptions of unrelated entities (e.g.
  *Köcherfliegen*, *Regenbremse*), so the suspected bio-feature bug
  becomes visible. The script MUST NOT silently swallow the text or
  short-circuit on missing context slots.

#### Scenario: Reh scenario reports PASS after the bio-feature fix

- **WHEN** the Reh scenario is executed against a `botCtl` that
  already incorporates the `fix-bio-feature-lookup` fix
- **THEN** the script prints a `PASS` line on stdout when
  `message.text` references `Reh` content AND does NOT mention
  `Köcherfliegen` or `Regenbremse`, and prints a `FAIL` line (with
  the offending substring) when the response still mixes in those
  unrelated species
