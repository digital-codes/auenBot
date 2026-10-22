#!/usr/bin/env python3
"""
botApiTest.py
=============

HTTP-driven test harness for the botCtl Flask service.

By default the script starts its own ``botCtl.py`` child process with a
generated (or ``--secret``-provided) ``STATE_TOKEN_SECRET``, waits for
the health endpoint, runs the requested scenario(s), and terminates the
child afterwards. With ``--no-start`` it attaches to an already-running
instance (secret via ``--secret`` or the environment).

The harness simulates the real frontend under the sealed-state-token
contract: it echoes the opaque ``state`` token from each response into
the next request, and uses the shared secret to decrypt the returned
token so context-level assertions (intent/entity/feature) keep working.

Quick start
-----------

    python botApiTest.py                        # spawn service + reh scenario
    python botApiTest.py --scenario all         # every scenario
    python botApiTest.py --no-start --secret <hex>   # attach to running service
    python botApiTest.py --url http://host:port

Scenarios
---------

    reh               reproduce the "was macht das reh" run (PASS/FAIL)
    frosch            plural query "gibt es da frösche" offers the five
                      frosch species, then "teichfrosch" completes
    entities          10 entity queries from the entity-resolution
                      contract (groups, singles, lookalikes, no-match)
    tampered          modified state token is rejected with 400
    foreign           session A's token is rejected in session B
    db_crosscheck     bot_history.db stores the decrypted plaintext context
    restart_resume    restart with same secret: session resumes
    restart_no_secret restart without secret: pre-restart token rejected
    all               run everything (restart scenarios run last)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

import botQa


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_URL = "http://127.0.0.1:11354/api"
HEALTH_PATH = "/api"  # relative to host; botCtl exposes GET /api
SERVICE_START_TIMEOUT = 300  # seconds; models load at service import
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_history.db")
SERVICE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botApiTest_service.log")


# ----------------------------------------------------------------------
# Service lifecycle
# ----------------------------------------------------------------------
class ServiceEnv:
    """Owns an optional ``botCtl.py`` child process and the shared secret.

    ``start()`` spawns the service with ``STATE_TOKEN_SECRET`` in its
    environment and polls the health endpoint until it responds.
    ``stop()`` terminates the child. ``restart(with_secret=...)`` stops
    and starts the child again, optionally *without* the secret in the
    environment (for the missing-secret restart scenario).
    """

    def __init__(self, url: str, secret: str, timeout: int = 30, verbose: bool = True):
        self.url = url if url.endswith("/api") else url.rstrip("/") + "/api"
        self.base_url = self.url[: -len("/api")] if self.url.endswith("/api") else url
        self.secret = secret  # str; botQa helpers receive secret.encode("utf-8")
        self.timeout = timeout
        self.verbose = verbose
        self.child: Optional[subprocess.Popen] = None
        self._log_fh = None

    # -- secret helpers -------------------------------------------------
    def seal(self, ctx: Dict[str, Any], session: str) -> str:
        token = botQa.seal_state(ctx, session, secret=self.secret.encode("utf-8"))
        if self.verbose:
            print(f"    seal : session={session[:8]}…  ctx={sorted(ctx.keys())}  "
                  f"-> {token[:12]}… ({len(token)} chars)")
        return token

    def open(self, token: str, session: str) -> Dict[str, Any]:
        ctx = botQa.open_state(token, session, secret=self.secret.encode("utf-8"))
        if self.verbose:
            print(f"    open : session={session[:8]}…  {token[:12]}… ({len(token)} chars)  "
                  f"-> ctx={sorted(ctx.keys())}")
        return ctx

    # -- process management ---------------------------------------------
    def is_healthy(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}{HEALTH_PATH}", timeout=self.timeout)
        except requests.RequestException:
            return False
        return r.status_code == 200 and r.json().get("status") == "ok"

    def _spawn(self, with_secret: bool) -> None:
        env = dict(os.environ)
        if with_secret:
            env["STATE_TOKEN_SECRET"] = self.secret
        else:
            env.pop("STATE_TOKEN_SECRET", None)
        self._log_fh = open(SERVICE_LOG, "a")
        self.child = subprocess.Popen(
            [sys.executable, "botCtl.py", "-d"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            stdout=self._log_fh,
            stderr=self._log_fh,
            stdin=subprocess.DEVNULL,
        )
        print(f"Started botCtl child (pid {self.child.pid}), "
              f"secret {'set' if with_secret else 'UNSET'}, "
              f"log: {SERVICE_LOG}")

    def wait_healthy(self, timeout: int = SERVICE_START_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_healthy():
                return True
            time.sleep(2)
        return False

    def start(self, with_secret: bool = True) -> bool:
        self._spawn(with_secret)
        if not self.wait_healthy():
            print(f"ERROR: service child did not become healthy within "
                  f"{SERVICE_START_TIMEOUT}s; see {SERVICE_LOG}", file=sys.stderr)
            return False
        return True

    def stop(self) -> None:
        if self.child is not None:
            self.child.terminate()
            try:
                self.child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=5)
            print("Stopped botCtl child.")
            self.child = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def restart(self, with_secret: bool = True) -> bool:
        self.stop()
        return self.start(with_secret=with_secret)


# ----------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------
class BotApiClient:
    """Thin wrapper around ``requests.Session`` for the botCtl API."""

    def __init__(self, url: str = DEFAULT_URL, timeout: int = 60):
        cleaned = url.rstrip("/")
        if not cleaned.endswith("/api"):
            cleaned = cleaned + "/api"
        self.url = cleaned
        self.base_url = cleaned[: -len("/api")]
        self.timeout = timeout
        self._session = requests.Session()

    def is_healthy(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}{HEALTH_PATH}", timeout=self.timeout)
        except requests.RequestException:
            return False
        return r.status_code == 200 and r.json().get("status") == "ok"

    def send_raw(self, payload: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
        """POST a payload, return (status_code, parsed json or None).

        Never raises for HTTP error statuses - error-path scenarios
        need the status code, not an exception.
        """
        r = self._session.post(self.url, json=payload, timeout=self.timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None


# ----------------------------------------------------------------------
# State propagation
# ----------------------------------------------------------------------
@dataclass
class ScenarioState:
    """Mutable per-scenario state carried across botCtl calls.

    On the wire only the opaque ``state`` token travels (plus session,
    sequence, lang, message). ``context()`` decrypts the current token
    with the harness's shared secret so scenarios can keep asserting on
    intent/entity/feature like they could under the old plaintext
    contract.
    """

    env: ServiceEnv = field(repr=False)
    session: str = ""
    sequence: int = 0
    lang: str = "de"
    state: str = ""
    last_message: Dict[str, Any] = field(default_factory=dict)

    def update(self, response: Dict[str, Any]) -> None:
        """Copy ``session``, ``sequence``, ``state``, and ``message``
        from a botCtl response onto this state.
        """
        if "session" in response and response["session"]:
            self.session = str(response["session"])
        if "sequence" in response and response["sequence"] is not None:
            try:
                self.sequence = int(response["sequence"])
            except (TypeError, ValueError):
                pass
        if isinstance(response.get("state"), str) and response["state"]:
            self.state = response["state"]
        if isinstance(response.get("message"), dict):
            self.last_message = dict(response["message"])

    def context(self) -> Dict[str, Any]:
        """Decrypt the current state token for assertions."""
        if not self.state:
            return {}
        return self.env.open(self.state, self.session)

    def payload(self, text: str, include_state: bool = True) -> Dict[str, Any]:
        """Build a request body for one user input."""
        payload: Dict[str, Any] = {
            "session": self.session,
            "sequence": self.sequence,
            "lang": self.lang,
            "message": {"text": text},
        }
        if include_state and self.state:
            payload["state"] = self.state
        return payload

    def turn(self, client: BotApiClient, text: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Send one input and update state from the response."""
        status, response = client.send_raw(self.payload(text))
        if self.env.verbose:
            print(f"    input: {text!r}  ->  HTTP {status}")
        if response is not None and status == 200:
            self.update(response)
            if self.env.verbose and self.state:
                ctx = self.context()
                opts = [o.get("title") if isinstance(o, dict) else o
                        for o in ctx.get("options", [])]
                if opts:
                    print(f"    options offered: {opts}")
                msg = self.last_message.get("text", "")
                msg = msg if isinstance(msg, str) else " | ".join(map(str, msg))
                print(f"    reply : {msg[:160]!r}{'…' if len(msg) > 160 else ''}")
        return status, response


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def run_scenario(client: BotApiClient, state: ScenarioState, inputs: List[str]) -> List[Dict[str, Any]]:
    """Drive ``inputs`` through ``state.turn`` in order. Returns the
    list of raw responses for inspection."""
    responses: List[Dict[str, Any]] = []
    for text in inputs:
        _, response = state.turn(client, text)
        responses.append(response or {})
    return responses


def _format_text(value: Any) -> str:
    """Pretty-print a ``message.text`` field that may be ``str`` or list."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n---\n".join(
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in value
        )
    return str(value)


def _flip_middle_char(token: str) -> str:
    """Flip one character in the middle of the token part.

    The token part is base64url; its LAST character may carry padding
    bits that decode to identical bytes, so flip a middle character -
    that always changes the ciphertext and must fail the auth tag.
    """
    head, tail = token[:3], token[3:]
    i = len(tail) // 2
    repl = "A" if tail[i] != "A" else "B"
    return head + tail[:i] + repl + tail[i + 1:]


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" - {detail}" if detail and not ok else ""))
    return ok


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
def scenario_reh(env: ServiceEnv) -> bool:
    """Reproduce the ``"was macht das reh"`` run; asserts on the
    decrypted context (white-box) and the bio-feature text contract."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state = ScenarioState(env=env)
    responses = run_scenario(client, state, ["was macht das reh"])
    response = responses[0] or {}

    ctx = state.context()
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

    leaking = [m for m in ("Köcherfliegen", "Regenbremse") if m in text]
    if leaking:
        ok = _report("reh", False, f"response mentions {leaking[0]}")
    elif not any(m in text for m in ("Ricke", "Kitz", "Rehbock", "Capreolus", "Geweih")):
        ok = _report("reh", False, "no Reh marker found")
    else:
        ok = _report("reh", True)
    print("=== end ===")
    return ok


def scenario_tampered_token(env: ServiceEnv) -> bool:
    """A modified state token must be rejected with HTTP 400."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state = ScenarioState(env=env)
    status, response = state.turn(client, "was macht das reh")
    if status != 200 or not state.state:
        return _report("tampered", False, f"setup turn failed: {status}")

    bad_token = _flip_middle_char(state.state)
    status, _ = client.send_raw({
        "session": state.session,
        "sequence": state.sequence,
        "lang": state.lang,
        "message": {"text": "weiter"},
        "state": bad_token,
    })
    print("=== Tampered token scenario ===")
    print(f"tampered token status: {status} (expected 400)")
    ok = _report("tampered", status == 400, f"expected 400, got {status}")
    print("=== end ===")
    return ok


def scenario_foreign_session(env: ServiceEnv) -> bool:
    """Session A's token must be rejected in a session B request."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state_a = ScenarioState(env=env)
    state_b = ScenarioState(env=env)

    status, _ = state_a.turn(client, "was macht das reh")
    if status != 200 or not state_a.state:
        return _report("foreign", False, f"session A setup failed: {status}")
    status, _ = state_b.turn(client, "hallo")
    if status != 200 or not state_b.state:
        return _report("foreign", False, f"session B setup failed: {status}")

    # send A's token in B's request
    status, _ = client.send_raw({
        "session": state_b.session,
        "sequence": state_b.sequence,
        "lang": state_b.lang,
        "message": {"text": "weiter"},
        "state": state_a.state,
    })
    print("=== Foreign session scenario ===")
    print(f"foreign token status: {status} (expected 400)")
    ok = _report("foreign", status == 400, f"expected 400, got {status}")
    print("=== end ===")
    return ok


def scenario_db_crosscheck(env: ServiceEnv) -> bool:
    """The latest RX/TX rows must store the decrypted plaintext context."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state = ScenarioState(env=env)

    # context the harness sent (decrypt the token it echoes) and the
    # context it received back
    sent_token_before = state.state
    status, _ = state.turn(client, "was macht das reh")
    if status != 200:
        return _report("db_crosscheck", False, f"turn failed: {status}")

    rx_expected = env.open(sent_token_before, state.session) if sent_token_before else {}
    tx_expected = state.context()

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT context FROM history WHERE session_id = ? AND message_type = 'RX' "
            "ORDER BY id DESC LIMIT 1", (state.session,))
        rx_row = cur.fetchone()
        cur.execute(
            "SELECT context FROM history WHERE session_id = ? AND message_type = 'TX' "
            "ORDER BY id DESC LIMIT 1", (state.session,))
        tx_row = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) FROM history WHERE session_id = ?", (state.session,))
        count = cur.fetchone()[0]
    finally:
        conn.close()

    print("=== DB crosscheck scenario ===")
    print(f"history rows for session: {count} (expected >= 2)")
    if rx_row is None or tx_row is None:
        return _report("db_crosscheck", False, "missing RX or TX row for session")

    rx_stored = json.loads(rx_row[0]) if rx_row[0] else {}
    tx_stored = json.loads(tx_row[0]) if tx_row[0] else {}

    ok = True
    if sent_token_before:
        if rx_stored != rx_expected:
            ok = _report("db_crosscheck", False,
                         f"RX context mismatch: stored={rx_stored} expected={rx_expected}")
    if tx_stored != tx_expected:
        ok = _report("db_crosscheck", False,
                     f"TX context mismatch: stored={tx_stored} expected={tx_expected}")
    if count < 2:
        ok = _report("db_crosscheck", False, f"expected >= 2 rows, got {count}")
    if ok:
        ok = _report("db_crosscheck", True)
    print("=== end ===")
    return ok


# ----------------------------------------------------------------------
# Entity-resolution samples (from the archived entity-resolution and
# bio-feature-lookup changes; expectations calibrated against the live
# service). Note: a completed turn consumes the entity requirement, so
# the final context has entity=None and the species is asserted on the
# answer text (which often pluralizes the name, e.g. "Teichfrösche").
# ----------------------------------------------------------------------
def _species_names() -> List[str]:
    """The 162 entity names from the live data file."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "entity_tp.json")) as f:
        return json.load(f)


FROSCH_SPECIES = ["Teichfrosch", "Springfrosch", "kleiner Wasserfrosch",
                  "Moorfrosch", "Laubfrosch"]
WINDROESCHEN_SPECIES = ["Gelbes Windröschen", "Busch-Windröschen"]
KNABENKRAUT_SPECIES = ["Helm-Knabenkraut", "Purpur-Knabenkraut"]
FLEDERMAUS_SPECIES = ["Wasserfledermaus", "Bechsteinfledermaus"]

ENTITY_SAMPLES: List[Dict[str, Any]] = [
    # --- Tiere (singular species -> direct completion naming it) ---
    {"name": "tier singular moorfrosch", "input": "moorfrosch",
     "text_contains": ["moorfrosch"]},
    # the standing bio-feature regression from the Reh log sample
    {"name": "tier reh bio-feature", "input": "was macht das reh",
     "text_any": ["ricke", "kitz", "rehbock", "capreolus", "geweih"],
     "text_not_contains": ["köcherfliegen", "regenbremse"]},
    {"name": "tier singular wildschwein", "input": "wildschwein",
     "text_contains": ["wildschwein"]},
    {"name": "tier plural fledermäuse + pick", "input": "fledermäuse",
     "options": FLEDERMAUS_SPECIES, "no_entity": True,
     "followup": "Wasserfledermaus", "text_contains": ["wasserfledermaus"]},
    # --- Pflanzen (groups use the options flow, singles complete) ---
    {"name": "pflanze singular scharbockskraut", "input": "scharbockskraut",
     "text_contains": ["scharbockskraut"]},
    {"name": "pflanze singular pfeifengras", "input": "pfeifengras",
     "text_contains": ["pfeifengras"]},
    # lookalike protection: windröschen must never resolve to frosch items
    {"name": "pflanze group windröschen + pick", "input": "windröschen",
     "options": WINDROESCHEN_SPECIES, "no_entity": True,
     "followup": "Busch-Windröschen", "accept_fallback": True,
     "text_not_contains": ["nichts gefunden"]},
    {"name": "pflanze group knabenkraut + pick", "input": "knabenkraut",
     "options": KNABENKRAUT_SPECIES, "no_entity": True,
     "followup": "Helm-Knabenkraut", "accept_fallback": True,
     "text_not_contains": ["nichts gefunden"]},
    # --- Auen (habitat explanation path, no species resolution) ---
    {"name": "aue rheinaue erklaerung", "input": "rheinaue",
     "text_contains": ["aue"], "no_species_options": True, "no_entity": True},
    # --- Auen (arbiter-recovered habitat entities; design D7 sets the
    # feature to 'definition' so the reply carries real content) ---
    {"name": "aue hartholzaue recovered", "input": "hartholzaue",
     "text_contains": ["hartholz"],
     "text_not_contains": ["nichts gefunden", "keine informationen"]},
    {"name": "aue stillgewaesser recovered", "input": "stillgewässer",
     "text_contains": ["stillgewässer"],
     "text_not_contains": ["nichts gefunden", "keine informationen"]},
    {"name": "aue feuchtwiesen recovered", "input": "feuchtwiesen",
     "text_contains": ["feuchtwiesen"],
     "text_not_contains": ["nichts gefunden", "keine informationen"]},
    {"name": "aue magerrasen recovered", "input": "magerrasen",
     "text_contains": ["magerrasen"],
     "text_not_contains": ["nichts gefunden", "keine informationen"]},
    # --- no-match: OOV term must not resolve to any species ---
    {"name": "oov luchs no species", "input": "luchs",
     "no_entity": True,
     "text_not_contains": ["otfuchs"]},
    # --- no-match: intent options offered, but never a species ---
    {"name": "no-match gibberish", "input": "xzqgrumpf blorps",
     "options_present": True, "no_species_options": True, "no_entity": True},
]


def scenario_frosch(env: ServiceEnv) -> bool:
    """The archived german-plural-entity-resolution acceptance flow:
    "gibt es da frösche" offers exactly the five frosch species with no
    entity filled; replying "teichfrosch" completes the flow."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state = ScenarioState(env=env)
    print("=== Frosch plural scenario ===")

    status, _ = state.turn(client, "gibt es da frösche")
    if status != 200:
        return _report("frosch", False, f"turn 1 failed: {status}")
    ctx = state.context()
    opts = [o.get("title") if isinstance(o, dict) else o for o in ctx.get("options", [])]
    print(f"    turn 1 options: {opts}")
    ok = True
    if sorted(map(str.lower, opts)) != sorted(map(str.lower, FROSCH_SPECIES)):
        ok = _report("frosch", False,
                     f"expected options {FROSCH_SPECIES}, got {opts}")
    if "entity" in ctx and ctx.get("entity"):
        ok = _report("frosch", False, f"entity filled on turn 1: {ctx.get('entity')!r}")

    status, _ = state.turn(client, "teichfrosch")
    if status != 200:
        return _report("frosch", False, f"turn 2 failed: {status}")
    text = _format_text(state.last_message.get("text")).lower()
    if "teichfrösch" not in text:
        ok = _report("frosch", False, f"turn 2 text does not mention Teichfrosch: {text[:120]!r}")
    if ok:
        ok = _report("frosch", True)
    print("=== end ===")
    return ok


def scenario_entity_samples(env: ServiceEnv) -> bool:
    """10 entity queries from the archived changes' contracts, each
    against a fresh session; every sample prints input, options, and
    the reply, then reports PASS/FAIL."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    species = {n.lower() for n in _species_names()}
    print(f"=== Entity samples scenario ({len(ENTITY_SAMPLES)} samples) ===")
    all_ok = True
    for i, sample in enumerate(ENTITY_SAMPLES, 1):
        name = sample["name"]
        state = ScenarioState(env=env)
        status, _ = state.turn(client, sample["input"])
        # Transient LLM timeouts (arbiter) can turn a resolvable query
        # into a no-match under suite load; retry once before failing.
        if status == 200 and (sample.get("text_contains") or sample.get("text_any")) \
                and "nichts gefunden" in _format_text(state.last_message.get("text")).lower():
            print("    transient no-match (likely arbiter timeout) - retrying once")
            state = ScenarioState(env=env)
            status, _ = state.turn(client, sample["input"])
        if status != 200:
            all_ok &= _report(f"sample {i}/{len(ENTITY_SAMPLES)} {name}", False,
                              f"turn failed: {status}")
            continue
        ctx = state.context()
        opts = [o.get("title") if isinstance(o, dict) else o for o in ctx.get("options", [])]
        text = _format_text(state.last_message.get("text"))
        low = text.lower()

        if sample.get("followup"):
            status, _ = state.turn(client, sample["followup"])
            if status != 200:
                all_ok &= _report(f"sample {i}/{len(ENTITY_SAMPLES)} {name}", False,
                                  f"followup failed: {status}")
                continue
            text = _format_text(state.last_message.get("text"))
            low = text.lower()
            # Feature fallback (conversation-recovery D8): when the picked
            # entity has no data for the requested feature, the reply
            # informs and offers up to 3 data-carrying features. Accept
            # that as the flow continuing - then pick one and expect real
            # content.
            if sample.get("accept_fallback") and "über einen dieser aspekte" in low:
                fopts = [o.get("title") if isinstance(o, dict) else o
                         for o in state.context().get("options", [])]
                print(f"    fallback options offered: {fopts}")
                if not fopts or len(fopts) > 3:
                    all_ok &= _report(f"sample {i}/{len(ENTITY_SAMPLES)} {name}", False,
                                      f"fallback options invalid: {fopts}")
                    continue
                status, _ = state.turn(client, str(fopts[0]))
                if status != 200:
                    all_ok &= _report(f"sample {i}/{len(ENTITY_SAMPLES)} {name}", False,
                                      f"fallback pick failed: {status}")
                    continue
                text = _format_text(state.last_message.get("text"))
                low = text.lower()

        ok = True
        detail = ""
        if "options" in sample:
            if sorted(map(str.lower, opts)) != sorted(map(str.lower, sample["options"])):
                ok, detail = False, f"expected options {sample['options']}, got {opts}"
        if sample.get("options_present") and not opts:
            ok, detail = False, "expected options to be offered, got none"
        if sample.get("no_species_options") and any(str(o).lower() in species for o in opts):
            ok, detail = False, f"species offered as options: {opts}"
        if sample.get("no_entity") and ctx.get("entity"):
            ok, detail = False, f"entity unexpectedly filled: {ctx.get('entity')!r}"
        for marker in sample.get("text_contains", []):
            if marker not in low:
                ok, detail = False, f"text missing {marker!r}: {low[:120]!r}"
        for marker in sample.get("text_not_contains", []):
            if marker in low:
                ok, detail = False, f"text mentions forbidden {marker!r}"
        if "text_any" in sample and not any(m in low for m in sample["text_any"]):
            ok, detail = False, f"text missing all of {sample['text_any']}"
        all_ok &= _report(f"sample {i}/{len(ENTITY_SAMPLES)} {name}", ok, detail)
    print("=== end ===")
    return all_ok


def scenario_breakout(env: ServiceEnv) -> bool:
    """Consecutive unresolvable inputs under an active intent must
    trigger the breakout: an inviting reply and a completely empty
    returned context. A successful turn first establishes the intent
    (deterministic path into the completion flow)."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state = ScenarioState(env=env)
    print("=== Breakout scenario ===")
    # establish a completed answer (focus + sticky intent)
    status, _ = state.turn(client, "was macht das reh")
    ctx = state.context()
    ok = status == 200 and ctx.get("focus", {}).get("entity") == "Reh"
    if not ok:
        return _report("breakout", False, f"setup turn failed: focus={ctx.get('focus')}")
    for i in range(1, 4):
        status, _ = state.turn(client, "xzqgrumpf blorps")
        ctx = state.context()
        print(f"    miss {i}: count={ctx.get('no_match_count')} "
              f"intent={'intent' in ctx} focus={'focus' in ctx}")
        if i < 3:
            ok &= status == 200 and "intent" in ctx
        else:
            t = _format_text(state.last_message.get("text")).lower()
            ok &= (status == 200 and ctx.get("no_match_count") == 0
                   and "intent" not in ctx and not ctx.get("focus")
                   and "von vorne" in t)
            if not ok:
                _report("breakout", False,
                        f"expected breakout on miss 3: count={ctx.get('no_match_count')}, "
                        f"reply={t[:80]!r}")
    # fresh conversation works afterwards
    status, _ = state.turn(client, "was macht das reh")
    t = _format_text(state.last_message.get("text"))
    ok &= status == 200 and ("Ricke" in t or "Kitz" in t or "Rehbock" in t)
    if ok:
        ok = _report("breakout", True)
    print("=== end ===")
    return ok


def scenario_followup(env: ServiceEnv) -> bool:
    """The focus-based follow-up: after a completed Reh answer, an
    entity-less feature question resolves against the focused entity.
    When the requested feature has no data, the feature-fallback options
    appear and picking one yields real content."""
    client = BotApiClient(url=env.url, timeout=env.timeout)
    state = ScenarioState(env=env)
    print("=== Follow-up scenario ===")
    status, _ = state.turn(client, "was macht das reh")
    ctx = state.context()
    ok = status == 200 and ctx.get("focus", {}).get("entity") == "Reh"
    if not ok:
        return _report("followup", False, f"setup turn failed: focus={ctx.get('focus')}")

    status, _ = state.turn(client, "und wie groß ist sie?")
    ctx = state.context()
    t = _format_text(state.last_message.get("text"))
    opts = [o.get("title") if isinstance(o, dict) else o for o in ctx.get("options", [])]
    print(f"    follow-up: options={opts} text[:90]={t[:90]!r}")
    if opts:
        # feature fallback offered - pick one, expect real content
        if len(opts) > 3:
            ok = _report("followup", False, f"more than 3 fallback options: {opts}")
            print("=== end ===")
            return ok
        status, _ = state.turn(client, str(opts[0]))
        t = _format_text(state.last_message.get("text"))
        print(f"    pick '{opts[0]}': text[:90]={t[:90]!r}")
        ok = (status == 200 and "keine Informationen" not in t
              and "Nichts gefunden" not in t and len(t) > 50)
    else:
        # direct content answer (sticky intent carries feature 'definition')
        ok = status == 200 and "Nichts gefunden" not in t and len(t) > 50 \
             and ("Ricke" in t or "kg" in t or "Kitz" in t)
    if ok:
        ok = _report("followup", True)
    print("=== end ===")
    return ok


def scenario_restart_same_secret(env: ServiceEnv) -> bool:
    """Restart with the same fixed secret: the session must resume."""
    url = env.url
    secret = os.urandom(16).hex()
    env = ServiceEnv(url=url, secret=secret)
    try:
        if not env.start(with_secret=True):
            return _report("restart_resume", False, "service did not start")
        client = BotApiClient(url=env.url, timeout=env.timeout)
        state = ScenarioState(env=env)
        status, _ = state.turn(client, "was macht das reh")
        if status != 200 or not state.state:
            return _report("restart_resume", False, f"pre-restart turn failed: {status}")
        ctx_before = state.context()
        token_before = state.state

        if not env.restart(with_secret=True):
            return _report("restart_resume", False, "service did not restart")

        status, response = state.turn(client, "weiter")
        ctx_after = state.context()
        print("=== Restart resume scenario ===")
        print(f"post-restart status: {status} (expected 200)")
        resumed = ctx_after.get("entity") == ctx_before.get("entity") \
            and ctx_after.get("intent") == ctx_before.get("intent")
        print(f"context resumed: {resumed} "
              f"(entity {ctx_before.get('entity')} -> {ctx_after.get('entity')})")
        ok = _report("restart_resume", status == 200 and resumed,
                     "session did not resume from the pre-restart token")
        print("=== end ===")
        return ok
    finally:
        env.stop()


def scenario_restart_missing_secret(env: ServiceEnv) -> bool:
    """Restart without STATE_TOKEN_SECRET: the pre-restart token must
    be rejected with HTTP 400."""
    url = env.url
    secret = os.urandom(16).hex()
    env = ServiceEnv(url=url, secret=secret)
    try:
        if not env.start(with_secret=True):
            return _report("restart_no_secret", False, "service did not start")
        client = BotApiClient(url=env.url, timeout=env.timeout)
        state = ScenarioState(env=env)
        status, _ = state.turn(client, "was macht das reh")
        if status != 200 or not state.state:
            return _report("restart_no_secret", False, f"pre-restart turn failed: {status}")
        token_before = state.state

        if not env.restart(with_secret=False):
            return _report("restart_no_secret", False, "service did not restart")

        status, _ = client.send_raw({
            "session": state.session,
            "sequence": state.sequence,
            "lang": state.lang,
            "message": {"text": "weiter"},
            "state": token_before,
        })
        print("=== Restart missing-secret scenario ===")
        print(f"pre-restart token after secret-less restart: {status} (expected 400)")
        ok = _report("restart_no_secret", status == 400, f"expected 400, got {status}")
        print("=== end ===")
        return ok
    finally:
        env.stop()


# name -> (runner, needs_exclusive_port)
SCENARIOS: Dict[str, Tuple[Any, bool]] = {
    "reh": (scenario_reh, False),
    "frosch": (scenario_frosch, False),
    "entities": (scenario_entity_samples, False),
    "breakout": (scenario_breakout, False),
    "followup": (scenario_followup, False),
    "tampered": (scenario_tampered_token, False),
    "foreign": (scenario_foreign_session, False),
    "db_crosscheck": (scenario_db_crosscheck, False),
    "restart_resume": (scenario_restart_same_secret, True),
    "restart_no_secret": (scenario_restart_missing_secret, True),
}
ALL_ORDER = ["reh", "frosch", "entities", "breakout", "followup", "tampered",
             "foreign", "db_crosscheck", "restart_resume", "restart_no_secret"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="botApiTest",
        description=(
            "Drive botCtl over HTTP. By default the harness starts its "
            "own botCtl.py child with a generated STATE_TOKEN_SECRET."
        ),
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Base URL of the botCtl /api endpoint (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--scenario", default="reh", choices=sorted(list(SCENARIOS.keys()) + ["all"]),
        help="Scenario to run (default: reh). 'all' runs every scenario.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="HTTP request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--secret", default=None,
        help="STATE_TOKEN_SECRET hex value (default: generate randomly).",
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--start", dest="start", action="store_true", default=True,
        help="Start the botCtl service as a child process (default).",
    )
    start_group.add_argument(
        "--no-start", dest="start", action="store_false",
        help="Attach to an already-running botCtl instead of spawning one.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    secret = args.secret or os.urandom(16).hex()

    env = ServiceEnv(url=args.url, secret=secret, timeout=args.timeout)
    attached = False

    if not args.start:
        if not env.is_healthy():
            print(f"ERROR: botCtl not reachable at {args.url} and --no-start given. "
                  "Start botCtl.py first (e.g. `STATE_TOKEN_SECRET=<hex> python botCtl.py -d`).",
                  file=sys.stderr)
            return 1
        attached = True
        print(f"Attaching to running botCtl at {args.url}.")
    else:
        if env.is_healthy():
            if not args.secret:
                print(f"ERROR: a botCtl is already running at {args.url} but no --secret "
                      "was given; the harness cannot decrypt its state tokens. "
                      "Pass the service's secret via --secret, or use --no-start.",
                      file=sys.stderr)
                return 1
            attached = True
            print(f"botCtl already running at {args.url}; attaching with provided secret.")
        else:
            if not env.start(with_secret=True):
                return 1

    selected = ALL_ORDER if args.scenario == "all" else [args.scenario]

    failed: List[str] = []
    try:
        for name in selected:
            fn, exclusive = SCENARIOS[name]
            if exclusive:
                if attached:
                    print(f"SKIP: {name} needs to control the service lifecycle "
                          "and cannot run in attach mode.", file=sys.stderr)
                    continue
                # free the port for the scenario's own child
                env.stop()
            if not fn(env):
                failed.append(name)
    finally:
        env.stop()

    if failed:
        print(f"\nFAILED scenarios: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nAll selected scenarios passed: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
