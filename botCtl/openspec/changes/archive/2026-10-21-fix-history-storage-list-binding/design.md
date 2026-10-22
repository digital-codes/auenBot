# Design

## Context

`botCtl.py` persists each conversation step via
`store_history()` into SQLite (`bot_history.db`) through
SQLAlchemy. The insert binds `message.get("text", None)` directly
as `message_text`. Upstream, `BotAction.get_entity_features()`
(`botActions.py`) returns its answer as
`{"text": [ ...strings... ], "image": [...], "audio": [...]}` —
including the one-element "no information available" sentinel —
and `BotIntent.execute()` passes that dict through as the output.
So TX messages with list-shaped `text` reach `store_history`
unchanged, SQLite rejects the list
(`Error binding parameter 9: type 'list' is not supported`, 1-based
index 9 = `message_text`), and the broad `except` in
`store_history` drops the row silently.

The pre-existing "FIXME merge output text array" block in
`route_handler` (lines ~486-494) was meant to handle these lists
but is broken: `message["text"][0].append(...)` calls `.append`
on a non-list element, and the `image`/`audio` branches call
`message["text"].append(...)` after `text` was already collapsed
to a scalar (or was never a list).

See proposal.md — Why for the full motivation and
specs/bot-history-storage for the behavioral contract.

## Goals / Non-Goals

**Goals:**
- Every RX/TX row lands in `bot_history.db`, including answers
  with list-shaped text.
- One normalization point for outgoing message text, so the API
  response and the stored row agree.
- Defense in depth at the storage boundary so a future producer
  of list/dict-shaped data degrades to JSON storage, not a
  dropped row.

**Non-Goals:**
- Changing the `history` table schema or migrating existing rows.
- Changing `BotAction.get_entity_features()`'s return shape — it
  is already specced (`bio-feature-lookup`: `text` is a list of
  strings) and other consumers rely on it.
- Reworking error handling of the API routes or removing the
  broad `except` in `store_history` beyond the logging it already
  does.

## Decisions

### D1: Normalize list text at the single TX assembly point in `route_handler`

Replace the broken FIXME block with a small, correct
normalization step applied to the intent execution result **before**
`store_history` and the `jsonify` response:

- `message["text"]` is a list → join entries with `"\n"` into one
  string (non-string entries are JSON-serialized first).
- `message["text"]` is missing/`None` → keep as `None`
  (`message_text` column is nullable).
- Otherwise (scalar str) → pass through unchanged.
- `image`/`audio` lists with more than one entry → keep the first
  entry and append a "more results" note, replacing the current
  broken branches. Lists with 0 or 1 entries pass through
  unchanged.

*Rationale:* one choke point guarantees the HTTP response and the
persisted row always carry the same normalized text, and it
removes the crashing code. *Alternative considered:* normalizing
inside `store_history` only — rejected as the sole fix because the
API response would still return a raw list where the `auenlaend`
app expects a string; and normalizing inside
`get_entity_features` — rejected because that changes the specced
action contract and its other consumers.

### D2: Coerce non-scalar column values in `store_history` (defense in depth)

Before constructing `HistoryRecord`, run each bound value through
a tiny helper: `str`/`int`/`float`/`bool`/`None`/`datetime` pass
through; anything else is `json.dumps(..., ensure_ascii=False)`'d.
This covers `message_text` and, cheaply, every other column
(e.g. `intent`, should a producer ever emit a list of intents).

*Rationale:* the spec requires that *any* payload the pipeline can
produce is storable; coercing only `message_text` would leave the
same failure mode one refactoring away. *Alternative:* SQLite
adapters / SQLAlchemy `TypeDecorator` — rejected as heavier
machinery than needed for one table and harder to see at the call
site.

### D3: Keep the join separator as `"\n"` (assumption recorded)

The proposal leaves the exact joined format open. `\n` keeps
multi-feature answers readable in both the chat UI and the stored
text; a JSON array string would preserve structure but leak
implementation formatting into user-facing text. Recorded as an
assumption; change is trivial if the app prefers another
separator.

### D4: Regression check via the existing HTTP harness

`botApiTest.py` already drives the live service and was used to
surface the previous bug. The Reh/bio-feature scenario doubles as
the regression check: after the fix it must complete with no
`Error storing history record` line in the service log and with
the TX row's `message_text` a string. Manual `sqlite3` /
SQLAlchemy inspection of the row is acceptable verification —
no new test framework is introduced.

## Risks / Trade-offs

- [Joining feature texts loses the list structure in storage]
  → The full unjoined message is still stored as JSON in the
  `message` column, so no information is lost for audit/debug.
- [Normalization touches the user-facing response] → The
  `auenlaend` app already receives a `message.text` that is
  sometimes a list today; a joined string is the shape the FIXME
  block was already attempting to produce, and the change is
  covered by the bot-api-testing harness scenarios.
- [`intent` or other columns could also arrive as lists from
  future producers] → Covered by the D2 blanket coercion rather
  than per-column special cases.
- [Broken image/audio "more results" branches] → Rewritten as
  part of D1; behavior for the 0/1-entry cases is unchanged, so
  no observable regression is expected.

## Migration Plan

No schema migration. Deploy is a restart of the Flask service
after the code change; rollback is reverting the single commit.
No data backfill: rows that failed to persist before the fix are
gone (they never reached the DB) and cannot be reconstructed.

## Open Questions

None.
