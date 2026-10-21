#!/usr/bin/env python3
"""
botApiTest.py
=============

HTTP-driven test harness for the botCtl Flask service.

This script drives a *running* botCtl.py instance (default URL
http://127.0.0.1:11354/api) the same way the real front-end does.
It does not boot botCtl.py itself - start botCtl.py first, then run
this script.

Quick start
-----------

    # In one terminal:
    python botCtl.py -d

    # In another:
    python botApiTest.py                       # runs the default Reh scenario
    python botApiTest.py --scenario reh        # explicit
    python botApiTest.py --url http://host:port

The first scenario reproduces the ``"was macht das reh"`` run that
appears in ``botCtl.log`` so the suspected bio-feature lookup bug in
``BotAction.find_entity`` becomes observable. More scenarios and
assertions will be added in follow-up change requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_URL = "http://127.0.0.1:11354/api"
HEALTH_PATH = "/api"  # relative to host; botCtl exposes GET /api


# ----------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------
class BotApiClient:
    """Thin wrapper around ``requests.Session`` for the botCtl API.

    Holds the configured base URL and exposes a small surface:
    ``health_check()`` to verify the server is reachable, and
    ``send()`` to POST a single user input against an existing
    scenario state.
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: int = 30):
        # Normalise the configured URL: callers may pass either the
        # full /api endpoint ("http://host:port/api") or just the
        # host ("http://host:port"). Append "/api" if missing so
        # health_check() and send() always hit the botCtl endpoint.
        cleaned = url.rstrip("/")
        if not cleaned.endswith("/api"):
            cleaned = cleaned + "/api"
        self.url = cleaned
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        """Return the host portion of ``url`` (without ``/api``)."""
        if self.url.endswith("/api"):
            return self.url[: -len("/api")]
        return self.url

    def health_check(self) -> bool:
        """Return ``True`` iff the botCtl ``GET /api`` health endpoint
        returns HTTP 200 with ``{"status": "ok"}``.
        """
        try:
            r = self._session.get(
                f"{self.base_url}{HEALTH_PATH}",
                timeout=self.timeout,
            )
        except requests.RequestException:
            return False
        if r.status_code != 200:
            return False
        try:
            payload = r.json()
        except ValueError:
            return False
        return payload.get("status") == "ok"

    def send(self, state: "ScenarioState", text: str) -> Dict[str, Any]:
        """POST a single user input to ``POST /api`` and return the
        parsed JSON response.

        ``state.session``, ``state.sequence``, and ``state.context``
        are used to build the request. The response itself is
        returned untouched - the caller is responsible for calling
        ``state.update(response)`` so the next ``send()`` carries the
        updated session/sequence/context forward.
        """
        payload: Dict[str, Any] = {
            "session": state.session,
            "sequence": state.sequence,
            "lang": state.lang,
            "message": {"text": text},
        }
        if state.context:
            payload["context"] = state.context

        r = self._session.post(
            self.url,
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


# ----------------------------------------------------------------------
# State propagation
# ----------------------------------------------------------------------
@dataclass
class ScenarioState:
    """Mutable per-scenario state carried across botCtl calls.

    The bot is stateful: ``session`` identifies the conversation,
    ``sequence`` increments per step, and ``context`` carries the
    resolved slots (``intent``, ``entity``, ``feature``, ``options``,
    ...). Each scenario holds its own ``ScenarioState`` and updates
    it from each response via ``update()`` so the next request sends
    the state the bot expects.
    """

    session: str = ""
    sequence: int = 0
    lang: str = "de"
    context: Dict[str, Any] = field(default_factory=dict)
    last_message: Dict[str, Any] = field(default_factory=dict)

    def update(self, response: Dict[str, Any]) -> None:
        """Copy ``session``, ``sequence``, ``context``, and
        ``message`` from a botCtl response onto this state.
        Missing fields are ignored so partial responses do not
        clobber valid state.
        """
        if "session" in response and response["session"]:
            self.session = str(response["session"])
        if "sequence" in response and response["sequence"] is not None:
            try:
                self.sequence = int(response["sequence"])
            except (TypeError, ValueError):
                pass
        if isinstance(response.get("context"), dict):
            self.context = dict(response["context"])
        if isinstance(response.get("message"), dict):
            self.last_message = dict(response["message"])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def run_scenario(
    client: BotApiClient,
    state: ScenarioState,
    inputs: List[str],
) -> List[Dict[str, Any]]:
    """Drive ``inputs`` through ``client.send`` in order, updating
    ``state`` after each call. Returns the list of raw responses
    so callers can inspect or assert on them.
    """
    responses: List[Dict[str, Any]] = []
    for text in inputs:
        response = client.send(state, text)
        state.update(response)
        responses.append(response)
    return responses


def _format_text(value: Any) -> str:
    """Pretty-print a ``message.text`` field that may be ``str``,
    ``list[str]``, or ``list[dict]`` (the botCtl code mixes all
    three). Always returns a string so it can be printed.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n---\n".join(parts)
    return str(value)


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
def scenario_reh(client: BotApiClient, state: Optional[ScenarioState] = None) -> List[Dict[str, Any]]:
    """Reproduce the ``"was macht das reh"`` run from ``botCtl.log``.

    Posts the question in German against a fresh session and prints
    the resolved intent/entity/feature plus the full ``message.text``.
    A PASS / FAIL line is emitted before the ``=== end ===`` marker so
    the bio-feature lookup regression is observable from the script's
    exit output.
    """
    state = state or ScenarioState()
    responses = run_scenario(client, state, ["was macht das reh"])
    response = responses[0]

    ctx = state.context or {}
    intent = ctx.get("intent")
    entity = ctx.get("entity")
    feature = ctx.get("feature")
    text = _format_text(state.last_message.get("text"))

    print("=== Reh scenario ===")
    print(f"intent  : {intent}")
    print(f"entity  : {entity}")
    print(f"feature : {feature}")
    print(f"session : {state.session}")
    print(f"sequence: {state.sequence}")
    if response.get("error"):
        print(f"error   : {response['error']}")
    print("--- message.text ---")
    print(text)
    print(_evaluate_reh_text(text))
    print("=== end ===")

    return responses


# Reh scenario bio-feature contract:
#   PASS  iff the text mentions at least one Reh marker
#         (Ricke / Kitz / Rehbock / Capreolus / Geweih)
#         AND mentions neither Köcherfliegen nor Regenbremse
#         (the entities that previously leaked through the
#         multi-entity find_entity bug).
#   FAIL  otherwise, naming the first offending substring.
REH_PASS_MARKERS = ("Ricke", "Kitz", "Rehbock", "Capreolus", "Geweih")
REH_FAIL_MARKERS = ("Köcherfliegen", "Regenbremse")


def _evaluate_reh_text(text: str) -> str:
    """Return ``"PASS"`` or ``"FAIL: <reason>"`` for the Reh scenario."""
    leaking = [m for m in REH_FAIL_MARKERS if m in text]
    if leaking:
        return f"FAIL: response mentions {leaking[0]}"
    if not any(m in text for m in REH_PASS_MARKERS):
        return "FAIL: no Reh marker found"
    return "PASS"


SCENARIOS = {
    "reh": scenario_reh,
}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="botApiTest",
        description=(
            "Drive a running botCtl instance over HTTP. "
            "Start botCtl.py first, then run this script."
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "Base URL of the botCtl /api endpoint "
            f"(default: {DEFAULT_URL})"
        ),
    )
    parser.add_argument(
        "--scenario",
        default="reh",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario to run (default: reh).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: 30).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    client = BotApiClient(url=args.url, timeout=args.timeout)

    if not client.health_check():
        print(
            f"ERROR: botCtl not reachable at {args.url}. "
            "Start botCtl.py first (e.g. `python botCtl.py -d`).",
            file=sys.stderr,
        )
        return 1

    scenario_fn = SCENARIOS.get(args.scenario)
    if scenario_fn is None:
        print(
            f"ERROR: unknown scenario '{args.scenario}'. "
            f"Available: {sorted(SCENARIOS.keys())}",
            file=sys.stderr,
        )
        return 2

    scenario_fn(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
