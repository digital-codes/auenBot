# Tasks

## 1. Scaffold the script and HTTP client

- [x] 1.1 Create `botApiTest.py` at the repository root with a
  `BotApiClient` class that wraps a `requests.Session` and exposes a
  `health_check()` method which calls `GET /api` on the configured
  URL and returns `True` only on HTTP 200. Verify by running
  `python botApiTest.py --help` and confirming the script parses
  without error.

- [x] 1.2 Add a `send(state, text)` method on `BotApiClient` that
  builds a payload shaped like `botSchema.json`
  (`input`, `session`, `sequence`, `lang`, `message.text`,
  `context`), POSTs it to `POST /api`, and returns the parsed JSON
  response. Verify by pointing the script at a temporarily running
  `botCtl.py` (e.g. `python botCtl.py -d &`) and confirming a single
  `health_check()` + one `send()` round-trip produces a 200 JSON
  response with `session` and `sequence` populated.

## 2. Add state propagation helper

- [x] 2.1 Add a `ScenarioState` dataclass to `botApiTest.py` that
  holds `session: str`, `sequence: int`, `context: dict`, and
  `last_message: dict`, plus a `update(response)` method that copies
  `session`, `sequence`, and `context` from a response. Verify by
  writing a one-shot script line that posts `"hallo"`, calls
  `state.update(response)`, and asserts `state.session` and
  `state.sequence` match the response fields.

- [x] 2.2 Add a `run_scenario(client, state, inputs)` helper that
  feeds a list of user-input strings through `client.send`, calling
  `state.update` after each call, and returns the list of responses.
  Verify by running the helper against the running `botCtl` with
  `["hallo auenbot", "tschüss"]` and confirming the second response
  echoes the same `session` as the first and that `sequence`
  increments.

## 3. Implement the Reh scenario

- [x] 3.1 Add a `scenario_reh(client, state)` function that posts
  `"was macht das reh"` with `lang="de"` against a fresh
  `ScenarioState`, prints the resolved `intent`, `entity`, and
  `feature` from the returned `context` (some of these may be
  `None` if the bot's `_finalize_completion` pops them after
  dispatch — see `botCtl.log` line 70), and prints the full
  `message.text`. Verify by running the function against a live
  `botCtl.py` and confirming the printed output includes
  `intent=tp_definition` and a non-empty `message.text` containing
  bio-feature content for the *Reh* query (the suspected bug's
  mixed-content output for *Köcherfliegen*/*Regenbremse* is fine —
  the goal is to *see* it, not to assert on it).

- [x] 3.2 Add an `argparse` CLI to `botApiTest.py` with `--url`
  (default `http://127.0.0.1:11354`) and `--scenario` (default
  `reh`) and call the matching scenario function from
  `if __name__ == "__main__"`. Verify by running
  `python botApiTest.py --url http://127.0.0.1:11354 --scenario reh`
  against a running `botCtl` and observing the same Reh output as
  in step 3.1, then running with `--scenario bogus` and confirming
  the script reports the unknown-scenario name cleanly.

## 4. Document and smoke-test

- [x] 4.1 Add a short header docstring to `botApiTest.py` that
  explains how to start `botCtl.py` first and how to invoke the
  Reh scenario. Verify by reading the docstring and confirming it
  mentions the default URL, the `reh` scenario, and that `botCtl`
  must already be running.

- [x] 4.2 Smoke-test the full flow end-to-end: start `botCtl.py`,
  run `python botApiTest.py`, and confirm the script exits 0 with
  the Reh context fields printed. Verify by capturing the exit
  code (`echo $?`) after the run.

## Workflow follow-up

- Open a follow-up change request that adds more scenarios (e.g.
  the *Frosch* case from `botCtl.log`) and asserts on the
  bio-feature text once the suspected `find_entity` bug is
  understood.
- Open a separate change request that fixes the suspected
  `BotAction.find_entity` bug and uses the script added here to
  verify the fix.
