"""
botMetis.py - Metis decision-model routing (add-metis-decision-routing).

Stage 1 decides the coarse intent (17-option choice) jointly with three
yes/no transition questions (followup, new_topic, unrelated) in one request.
Stage 2 decides the bio feature category (verhalten/eigenschaften/beschreibung).
No text is generated; the endpoint returns probabilities only.

Disabled or failing Metis routing lets the caller fall back to the legacy
embedding decoder (design D1/D8 of the change).
"""
import json
import os

import requests

# Routing thresholds, spike-validated (batch over 882 real example
# questions, iterations 1-4 in /tmp/opencode/metis_spike/).
P_SOCIAL = 0.5        # min probability for smalltalk/beleidigung to win
UNRELATED = 0.8       # noul threshold for the fallback route
FOLLOWUP = 0.7        # noul threshold to reuse the focus entity
NEW_TOPIC = 0.7       # noul threshold to clear the focus
TIE_MARGIN = 0.10     # stage-2 undecided: top-2 probability margin
TIE_CONFIDENCE = 0.35  # stage-2 undecided: min confidence

# Stage-1 intent name -> intent name executed by BotIntent (existing
# records; only bio_frage is a new record appended at load time).
INTENT_ROUTING = {
    "smalltalk": "begrüßung",
    "beleidigung": "ärgerlich",
    "bio_frage": "bio_frage",
    "nazka_info": "naturschutzzentrum",
    "anreise": "anreise",
    "ausstellungen": "ausstellungen",
    "kinder": "kinder",
    "aktivitaeten": "angebote",
    "essen": "verpflegung",
    "hunde": "hunde in der ausstellung",
    "fuehrungen": "führungen",
    "barrierefreiheit": "barrierefreiheit",
    "toiletten": "toiletten",
    "wetter": "wetter",
    "messdaten": "messdaten",
    "expertise": "expertise",
    "fallback": "fallback",
}

SOCIAL_INTENTS = ("smalltalk", "beleidigung")
CATEGORIES = ("verhalten", "eigenschaften", "beschreibung")


def load_private():
    """Read the METIS_* config from private.py (optional file)."""
    try:
        import private as pr  # type: ignore
    except Exception:
        return None
    return {
        "key": getattr(pr, "METIS_KEY", None),
        "url": getattr(pr, "METIS_URL", "https://api.greenpt.ai/v1/systemone"),
        "model": getattr(pr, "METIS_MDL", "metis"),
        "enabled": getattr(pr, "METIS_ROUTING", None),
    }


class MetisRouter:
    """Thin client for POST /v1/systemone with the two routing stages."""

    def __init__(self, config=None, data_dir=None):
        cfg = config or load_private()
        self.key = cfg.get("key") if cfg else None
        self.url = (cfg.get("url") if cfg else None) or "https://api.greenpt.ai/v1/systemone"
        self.model = (cfg.get("model") if cfg else None) or "metis"
        flag = cfg.get("enabled") if cfg else None
        # Default on when a key is present; METIS_ROUTING=false overrides.
        self._enabled = bool(self.key) if flag is None else bool(flag) and bool(self.key)
        self.DEBUG = False

        data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.criteria = {}
        self.categories = {}
        try:
            with open(os.path.join(data_dir, "intents_metis.json"), encoding="utf-8") as f:
                self.criteria = json.load(f).get("criteria", {})
            with open(os.path.join(data_dir, "category_map.json"), encoding="utf-8") as f:
                self.categories = json.load(f)
        except Exception as e:
            if self.DEBUG:
                print("MetisRouter: data files not loaded:", e)

    def setDebug(self, debug):
        self.DEBUG = debug

    def enabled(self):
        return self._enabled and bool(self.criteria) and bool(self.categories)

    # ------------------------------------------------------------------
    def _ask(self, state, questions):
        """One request; one retry on 429/5xx honouring Retry-After (D8)."""
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        body = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(2):
            r = requests.post(self.url, json=body, headers=headers, timeout=60)
            if r.status_code in (429,) or r.status_code >= 500:
                if attempt == 0:
                    import time
                    time.sleep(min(int(r.headers.get("Retry-After", 3)), 10))
                    continue
            r.raise_for_status()
            return r.json()

    def detect(self, query, focus_entity=None, focus_type=None, prev_answered=False):
        """Stage 1: intent choice + followup/new_topic/unrelated, jointly."""
        state = {
            "nutzer_eingabe": query,
            "vorherige_frage_beantwortet": bool(prev_answered),
            "fokus_lebewesen": f"{focus_entity} ({focus_type})" if focus_entity else None,
        }
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Was will der Nutzer?",
                "criteria": self.criteria,
            },
            "followup": {
                "type": "noul",
                "instructions": "Ist diese Eingabe eine Folgefrage zu derselben Person, Pflanze oder Aue wie in der vorherigen Frage, die bereits vollstaendig beantwortet wurde?",
            },
            "new_topic": {
                "type": "noul",
                "instructions": "Beginnt der Nutzer hier ein neues Thema und fragt nach einer anderen Person, Pflanze oder Aue als in der vorherigen Frage?",
            },
            "unrelated": {
                "type": "noul",
                "instructions": "Hat die Eingabe nichts mit Tieren, Pflanzen, Auen oder dem Naturschutzzentrum und alles, was darum herum gehoert, zu tun und ist auch keine Begruessung oder Hoeflichkeit?",
            },
        }
        answers = self._ask(state, questions)["answers"]
        return {
            "intent": answers["intent"]["choice"],
            "probs": answers["intent"]["probabilities"],
            "followup": answers["followup"]["noul"],
            "new_topic": answers["new_topic"]["noul"],
            "unrelated": answers["unrelated"]["noul"],
        }

    def category(self, query, entity_name):
        """Stage 2: feature category for a resolved entity (D3)."""
        state = {
            "nutzer_eingabe": query,
            "lebewesen": entity_name,
        }
        questions = {
            "category": {
                "type": "choice",
                "instructions": "Welche Art von Information will der Nutzer ueber das Lebewesen oder die Aue wissen?",
                "criteria": {cid: c["criteria"] for cid, c in self.categories.items()},
            }
        }
        answers = self._ask(state, questions)["answers"]["category"]
        probs = answers["probabilities"]
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        p1, p2 = ranked[0][1], ranked[1][1]
        tie = (p1 - p2) < TIE_MARGIN or answers.get("confidence", 1.0) < TIE_CONFIDENCE
        return {
            "category": answers["choice"],
            "tie": tie,
            "probs": probs,
            "confidence": answers.get("confidence"),
        }
