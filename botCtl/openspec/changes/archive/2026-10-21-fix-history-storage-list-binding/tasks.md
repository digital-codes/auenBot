# Tasks

## 1. Normalize outgoing TX message in `botCtl.py`

- [x] 1.1 Replace the broken "FIXME merge output text array" block in `route_handler` with a normalization step that joins a list-shaped `message["text"]` into one `"\n"`-joined string (non-string entries JSON-serialized first), passes scalar `text` through unchanged, and keeps `text` absent/`None` as `None`. Verify with a quick REPL check: `{"text": ["a", "b"]}` → `"a\nb"`, `{"text": "a"}` → `"a"`, `{}` → no error.
- [x] 1.2 In the same normalization step, fix the `image`/`audio` handling: when more than one entry, keep the first entry and append the "more results" note to the (now string) `text`; with 0 or 1 entries leave the value unchanged. Verify by constructing an intent result with `text` as a list plus 2 image URLs and confirming no `AttributeError` and the expected single-image output.
- [x] 1.3 Confirm the normalized `message` is what both `store_history` and the `jsonify` response receive (same dict flows to both). Verify by reading the code path and adding a temporary DEBUG print of the normalized text if needed.

## 2. Harden `store_history` against unsupported types

- [x] 2.1 Add a small coercion helper in `botCtl.py` that passes `str`/`int`/`float`/`bool`/`None`/`datetime` through unchanged and JSON-serializes anything else (`json.dumps(..., ensure_ascii=False)`). Verify with a REPL check: list, dict, tuple → JSON strings; `"x"`, `1`, `None`, `datetime` → unchanged.
- [x] 2.2 Apply the helper to the values bound in `HistoryRecord(...)` inside `store_history` (at minimum `message_text`, blanket-applied to the other scalar columns per design D2). Verify by calling `store_history` directly with `message={"text": ["a", "b"]}` and confirming a row is committed and `message_text == "a\nb"` (see 3.1).

## 3. Verification against the live service and database

- [x] 3.1 Write and run a direct DB check: call `store_history` with a list-shaped `message["text"]` and a dict-valued extra, then query the newest `history` row (sqlite3 CLI or a short SQLAlchemy snippet) and assert `message_text` is `str`, the JSON columns parse, and no exception was logged. Verification: query returns the expected row.
- [x] 3.2 Start the Flask service and run the bio-feature scenario from `botApiTest.py` (e.g. the *Reh* request). Verification: the response's `message.text` is a single string, and the service log contains no `Error storing history record` line for the run.
- [x] 3.3 Inspect the TX row created by 3.2 in `bot_history.db`: `message_type = "TX"`, `message_text` is a `str` containing the feature text, `message` column still holds the full JSON message. Verification: one SQL query shows all three properties.

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Verify the archived result.
