# Proposal

## Why

Every conversation step is supposed to be persisted to SQLite
(`bot_history.db`), but TX rows whose output text is a list never
reach the database: the log shows
`Error storing history record: (sqlite3.ProgrammingError) Error binding parameter 9: type 'list' is not supported`.
Parameter 9 of the `history` insert is `message_text`, and
`store_history` in `botCtl.py` binds `message.get("text", None)`
raw. The bio-feature action (`BotAction.get_entity_features` in
`botActions.py`) legitimately returns `text` as a **list** of
feature strings (and a sentinel list when nothing was found), so
whenever that shape flows through `BotIntent.execute` into
`store_history`, the insert fails, the row is silently dropped by
the broad `except`, and the conversation history is incomplete.

The existing handling for list outputs in `botCtl.py` (the
"FIXME merge output text array" block, lines ~486-494) is itself
broken: it calls `.append` on the first list element (a dict or
str, not a list), mutates `message["text"].append(...)` only in
branches that were already reduced to a single item, and leaves
the common case — a list of plain strings — untouched. So the bug
is observable twice: a broken merge attempt before storage and a
failed insert at storage time.

## What Changes

- Normalize the intent execution output in `botCtl.py` so that
  `message["text"]` is always a single string before it is stored
  or returned: a list of strings SHALL be joined into one string;
  a single non-list value SHALL pass through unchanged.
- Fix (replacing) the broken "merge output text array" block so it
  no longer calls `.append` on non-list values and no longer
  crashes when `text`, `image`, or `audio` arrive as lists.
- Harden `store_history` so no column binding can receive an
  unsupported type again: `message_text` (and any other value that
  is not `str`/`int`/`float`/`bool`/`None`) SHALL be coerced to a
  JSON string before insert, so a future producer of list-shaped
  data degrades to a stored JSON array instead of a dropped row.
- Keep the SQLite schema unchanged — this is a write-path fix, no
  migration is needed.

## Capabilities

### New Capabilities

- `bot-history-storage`: Capability that owns the contract for
  persisting every conversation step (RX and TX) to the `history`
  table: which columns are written, and the guarantee that any
  message payload the pipeline can produce is storable — a row
  MUST NOT be silently dropped because a field arrived as a list
  or dict.

### Modified Capabilities

(None. `bio-feature-lookup` already specifies that the action's
`text` field is a list of strings; the defect is in the caller's
storage path, which is owned by the new `bot-history-storage`
capability. No existing requirement changes.)

## Impact

- `botCtl.py`: `store_history` (defensive coercion) and the TX
  output normalization block in `route_handler`. No route, schema,
  or API contract changes — responses keep the same keys.
- `botActions.py`: none required; its list-shaped `text` output is
  documented as valid input for the fix.
- `bot_history.db`: no schema change; existing rows are unaffected.
- Regression check via the existing `botApiTest.py` harness: a
  bio-feature scenario (e.g. *Reh*) must produce one stored TX row
  with a string `message_text` and no `Error storing history
  record` line in the log.
