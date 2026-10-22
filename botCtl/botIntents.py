import json
import os
import re

try:
    from rapidfuzz import process, fuzz, utils
    matchProcessor = utils.default_process
except ImportError:
    # ubuntu 20.04 LTS compatibility
    from fuzzywuzzy import process, fuzz, utils
    matchProcessor = utils.full_process

import random

import botText
from botText import german_stem
from botActions import BotAction
from botLlm import OpenAICompatClient


# Context schema (shared static JSON) ------------------------------------------------
DEFAULT_CONTEXT_SCHEMA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), ".", "botSchema.json")
)

def load_context_schema(path=None):
    """Load and return the context JSON schema from path or default location.
    Returns an empty dict if the file is missing. Raises on JSON parse errors.
    """
    p = path or DEFAULT_CONTEXT_SCHEMA_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            schema = json.load(f)
        context = schema.get("properties", {}).get("context", {}).get("properties", {})
        return context
        
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"Error loading context schema '{p}': {e}")

# module-level schema object for import by other modules
CONTEXT_SCHEMA = load_context_schema()


INTENT_MESSAGES = {
    "de": {
        "no_intent": "Kein Intent im Kontext angegeben.",
        "intent_not_found": "Intent nicht gefunden.",
        "handler_not_found": "Handler nicht gefunden.",
        "options_file_not_found": "Fehler: Optionsdatei für Completion nicht gefunden.",
        "no_input_completion": "Fehler: Kein Eingabewert für Completion.",
        "no_items_file": "Fehler: Keine Items-Datei für Completion-Intent vorhanden.",
        "no_match_completion": "Das habe ich leider nicht verstanden. Kannst du es anders formulieren?",
        "no_input_options": "Fehler: Kontext hat Optionen, aber keine Eingabe angegeben.",
        "invalid_key": "Fehler: Ungültiger Schlüssel.",
        "select_options": "Bitte wählen Sie eine der folgenden Optionen:",
        "completed": "Abgeschlossen.",
        "continue_completion": "Bitte fahren Sie mit der Vervollständigung fort."
    },
    "en": {
        "no_intent": "No intent specified in context.",
        "intent_not_found": "Intent not found.",
        "handler_not_found": "Handler not found.",
        "options_file_not_found": "Error: Options file not found for completion.",
        "no_input_completion": "Error: No input for completion.",
        "no_items_file": "Error: No items file available for completion intent.",
        "no_match_completion": "Sorry, I didn't understand that. Could you rephrase it?",
        "no_input_options": "Error: Context has options but no input provided.",
        "invalid_key": "Error: Invalid key.",
        "select_options": "Please select one of the following options:",
        "completed": "Completed.",
        "continue_completion": "Please continue with the completion."
    },
    "fr": {
        "no_intent": "Aucune intention spécifiée dans le contexte.",
        "intent_not_found": "Intention non trouvée.",
        "handler_not_found": "Gestionnaire non trouvé.",
        "options_file_not_found": "Erreur : Fichier d'options introuvable pour la complétion.",
        "no_input_completion": "Erreur : Aucune entrée pour la complétion.",
        "no_items_file": "Erreur : Aucun fichier d'éléments disponible pour l'intention de complétion.",
        "no_match_completion": "Désolé, je n'ai pas compris. Peux-tu le reformuler ?",
        "no_input_options": "Erreur : Le contexte a des options mais aucune entrée fournie.",
        "invalid_key": "Erreur : Clé invalide.",
        "select_options": "Veuillez sélectionner l'une des options suivantes :",
        "completed": "Terminé.",
        "continue_completion": "Veuillez continuer avec la complétion."
    },
}


__all__ = ["CONTEXT_SCHEMA", "load_context_schema", "DEFAULT_CONTEXT_SCHEMA_PATH", "INTENT_MESSAGES"]

class BotIntent:
    """
    BotIntent
    ----------
    Loads a JSON intent definition file and provides simple handlers to execute
    intents. The class is small and intentionally conservative: it expects a file
    that contains either a list of intent dictionaries or a dictionary of
    dictionaries (values are intent dicts). A single intent dict is also
    accepted.

    Usage:
      intent = BotIntent("/path/to/intents.json")
      intent.setDebug(True)
      result = intent.execute("default", intent_name="greet", context={}, input="hi")

    Public methods:
      - setDebug(bool)
      - get_intent_by_id(id) -> dict|None
      - get_intent_by_name(name) -> dict|None
      - execute(handler="default", intent_name=None, input=None, context=None, lang="de")
    """

    def __init__(self, path, parameters=None):
        self.DEBUG = False
        self.path = path
        self.name = os.path.basename(path)
        self.parameters = parameters or {}
        self.actionCtl = None
        self.actionNames = []
        self.matchThreshold = 75  # default fuzzy match threshold
        self.maxNoMatch = 3      # breakout limit for consecutive no-match turns
        self.llm = None           # LLM client for the entity arbiter
        self._domain_lists_cache = None  # cached arbiter candidate lists

        # load data file (accept list, dict-of-dicts or single dict)
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, list):
                self.data = loaded
            elif isinstance(loaded, dict):
                # if values are dicts, treat it as a dict-of-records
                if all(isinstance(v, dict) for v in loaded.values()):
                    self.data = list(loaded.values())
                else:
                    # fallback: wrap single dict as single record list
                    self.data = [loaded]
            else:
                self.data = []

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{self.name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}")

        # named handlers available to execute()
        self.handlers = {
            "default": self.__defaultHandler,
            "complete": self.__completeHandler,
        }

    @staticmethod
    def getMsg(key, lang = "de"):
        """Get a localized message by language code and key."""
        if not lang or lang not in INTENT_MESSAGES:
            lang = "de"
        msgs = INTENT_MESSAGES.get(lang, INTENT_MESSAGES["de"])
        return msgs.get(key, f"{msgs.get('invalid_key', '')} ({key})")

    @staticmethod
    def dummyAction(name, input=None, lang="de"):
        """A dummy action function that can be used as a handler."""
        if name is None or name.strip() == "":
            return {"text": "Error: Unnamed actions cannot be called!"}
        output = {}
        if input is None or input.strip() == "":
            output["text"] = f"Dummy action '{name}' executed with no input."
        else:
            opts = random.sample(["text","link","options"], k=random.randint(1,3))
            input = input.strip() + " " + " ".join(opts)
            if "text" in input.lower():
                output["text"] = f"Dummy action '{name}' received text input: {input}"
            if "link" in input.lower():
                output["link"] = {"title":"Testlink", "url":f"http://example.com/dummy_action/{name}"}
            if "options" in input.lower():
                output["options"] = [
                    {"title": f"Option 1 for {name}", "value": "opt1"},
                    {"title": f"Option 2 for {name}", "value": "opt2"},
                    {"title": f"Option 3 for {name}", "value": "opt3"},
                ]
        return output

    def setDebug(self, debug: bool):
        """Enable or disable debug printing."""
        self.DEBUG = bool(debug)
        if self.actionCtl is not None:
            self.actionCtl.setDebug(self.DEBUG)

    def setThreshold(self, threshold: int):
        """Set the fuzzy match threshold (0-100)."""
        self.matchThreshold = int(threshold)
        if self.actionCtl is not None:
            self.actionCtl.setThreshold(self.matchThreshold)

    def setLlm(self, llm):
        """Inject the LLM client used by the entity arbiter.

        When not set, the arbiter lazily constructs a client from the
        private config (same pattern as botCtl.py / botDecoder.py).
        """
        self.llm = llm

    def getThreshold(self):
        """Get the fuzzy match threshold (0-100)."""
        return self.matchThreshold

    def setMaxNoMatch(self, n: int):
        """Set the breakout limit for consecutive no-match turns."""
        self.maxNoMatch = int(n)

    def getMaxNoMatch(self) -> int:
        """Get the breakout limit for consecutive no-match turns."""
        return self.maxNoMatch

    def setActions(self, definition_file):
        """Set a BotActions instance to use for action processing."""
        self.actionCtl = BotAction(definition_file)
        self.actionNames = self.actionCtl.getHandlers()
        if self.DEBUG:
            print(f"BotIntent: Loaded actions from '{definition_file}':", self.actionNames)

    def get_intent_by_id(self, intent_id):
        """Return the intent dict with matching 'id' or None if not found."""
        for entry in self.data:
            if entry.get("id") == intent_id:
                return entry
        return None

    def get_intent_by_name(self, intent_name):
        """Return the intent dict with matching 'intent' or None if not found."""
        for entry in self.data:
            if entry.get("intent") == intent_name:
                return entry
        return None

    def execute(self, intent_name=None, input=None, context=None, lang="de"):
        """
        Execute the handler of an intent.

        - intent_name: name of intent to execute (must be present in loaded data)
        - input: user input string (may be None)
        - context: dict representing current context; if None an empty dict is used
        - lang: language suffix used to pick utterances (e.g. 'de' -> 'utter_de')

        Returns: dict with keys "output" and "context".
        """
        if self.DEBUG:
            print(f"Executing handler for intent '{intent_name}'")

        if intent_name is None or intent_name.strip() == "":
            raise ValueError("No intent name specified for execution.")
        intent_name = intent_name.strip()
        intent = self.get_intent_by_name(intent_name)
        if intent is None:
            raise ValueError(f"Intent '{intent_name}' not found.")
        handler = intent.get("handler", "default")
        if self.DEBUG:
            print(f"Using handler '{handler}' for intent '{intent_name}'")

        # resolve handler (accept callables for flexibility)
        if callable(handler):
            handler_func = handler
        else:
            handler_func = self.handlers.get(handler)
            if handler_func is None:
                raise ValueError(f"Handler '{handler}' not found.")

        # find intent
        intent = self.get_intent_by_name(intent_name)
        if self.DEBUG:
            print(f"Resolved intent for '{intent_name}':", intent)
        if intent is None:
            raise ValueError(f"Intent '{intent_name}' not found.")

        # ensure we don't accidentally mutate a caller's context reference
        ctx = dict(context) if context is not None else {}
        result = handler_func(intent, input=input, context=ctx, lang=lang)

        if self.DEBUG:
            print(f"Handler '{handler}' result:", result)
        return result

    def _make_error_response(self, msg_key, context, intent, lang):
        """
        Create an error response with localized message.
        
        Args:
            msg_key: Message key from INTENT_MESSAGES
            context: Current context dict
            intent: Intent dict
            lang: Language code
            
        Returns:
            dict with "output" and "context" keys
        """
        if self.DEBUG:
            print(f"_make_error_response: Creating error response for key '{msg_key}'")
        message = BotIntent.getMsg(msg_key, lang)
        # always remove options and target on error. leave rest
        # add error info to context?
        ctx = dict(context)
        ctx.pop("options", None)
        ctx.pop("target", None)
        #ctx["last_intent"] = intent.get("intent")
        #ctx["intent"] = None
        return {"output": {"text": message}, "context": ctx, "error": msg_key}

    def _load_options_file(self, option_name):
        """
        Load options from data/options_{option_name}.json.
        
        Args:
            option_name: Name of the options file (without prefix/suffix)
            
        Returns:
            List of option dicts or None if file not found
        """
        if self.DEBUG:
            print(f"_load_options_file: Loading options for '{option_name}'")
        try:
            options_path = os.path.join(os.path.dirname(__file__), "data", f"options_{option_name}.json")
            with open(options_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.DEBUG:
                print(f"Options file not found for '{option_name}'")
            return None

    def _load_items_file(self, requirement, intent_base):
        """
        Load items from data/{requirement}_{intent_base}.json.
        
        Args:
            requirement: Requirement name
            intent_base: Base name of the intent
            
        Returns:
            List of items or None if file not found
        """
        if self.DEBUG:
            print(f"_load_items_file: Loading items for requirement '{requirement}' and intent base '{intent_base}'")
        try:
            items_path = os.path.join(os.path.dirname(__file__), "data", f"{requirement}_{intent_base}.json")
            with open(items_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.DEBUG:
                print(f"Items file not found for requirement '{requirement}' and intent base '{intent_base}'")
            return None

    def _find_first_missing_requirement(self, requirements, context):
        """
        Find the first requirement that is not present in context.
        
        Args:
            requirements: List of requirement names
            context: Current context dict
            
        Returns:
            First missing requirement name or None if all satisfied
        """
        if self.DEBUG:
            print(f"_find_first_missing_requirement: Checking requirements {requirements} against context keys {list(context.keys())}")
        for req in requirements:
            if req not in context:
                return req
        return None

    def _match_input_to_items(self, input_text, items):
        """
        Match user input against a list of items using fuzzy matching.
        
        Args:
            input_text: User input string
            items: List of item strings
            threshold: Minimum score for a valid match (default: 75)
            
        Returns:
            Matched item string or None if no match above threshold
        """
        if self.DEBUG:
            print(f"_match_input_to_items: Matching input '{input_text}' against items")
        choices = [str(i) for i in items]
        if self.DEBUG:
            print(f"Matching input '{input_text}' against items (count={len(choices)})")

        match = process.extractOne(input_text.lower(), [c.lower() for c in choices], 
                                   scorer=fuzz.WRatio, processor=matchProcessor)
        if not match:
            if self.DEBUG:
                print("No match found (extractOne returned None).")
            return None

        # fuzzywuzzy returns (match, score), rapidfuzz returns (match, score, index)
        if len(match) > 2:
            _, score, idx = match
            term = choices[idx]
        else:
            term_, score = match
            term = [t for t in choices if t.lower() == term_.lower()][0]
            
        if self.DEBUG:
            print("Match result:", match, term)
        
        if score >= self.matchThreshold:
            return term
        
        if self.DEBUG:
            print(f"Input '{input_text}' did not match any item (score {score}).")
        return None

    def _match_input_to_items_structured(self, input_text, items):
        """
        Staged entity matching per design D2 (german-plural-entity-resolution).

        Stage 1 - exact stem containment: stem both the input tokens
        (botText.normalize) and the item names (full name and each
        word), and collect items whose name stem equals an input
        token stem, or whose word stems start/end with it (German
        compound handling, e.g. 'frosch' in 'Teichfrosch').

        Stage 2 - fuzzy fallback (only when stage 1 found nothing):
        the existing WRatio match, but per input token against the
        item names instead of the whole sentence, accepted at
        self.matchThreshold.

        Returns a structured result dict or None:
          {"kind": "single", "name": str}
          {"kind": "group", "names": [str, ...]}  # ordered best-first
        """
        if self.DEBUG:
            print(f"_match_input_to_items_structured: Matching '{input_text}' against {len(items)} items (staged)")

        input_stems = [s for s in botText.normalize(input_text) if len(s) >= 3]
        if not input_stems:
            return None

        # Precompute stems for every item: full name, each word's
        # stem, and each raw lowercase word (containment candidates).
        item_stems = []  # list of (item, full_stem, word_stems, raw_words)
        for item in items:
            name = str(item)
            full_stem = german_stem(name)
            words = [w.lower() for w in re.split(r"[^A-Za-zÄÖÜäöüß]+", name) if w]
            word_stems = [german_stem(w) for w in words]
            item_stems.append((name, full_stem, word_stems, words))

        # Stage 1: exact stem relation, ranked by match quality.
        # Loose prefix/suffix containment (rank 2) demands at least a
        # 4-char token stem that is not a function word, so short
        # words ("wie", "das", "gibt") cannot spuriously match item
        # stems ("Wiesen-...") while genuine short terms still match.
        # 4 chars (not 3): 3-char stems like "gro" (from "groß")
        # prefix-match a dozen "Groß-*"/"Große-*" entities at once and
        # hijack turns as bogus group matches (design D9 of
        # add-no-match-breakout-and-focus-followups).
        #
        # Rank-1 word-stem hits distinguish the FINAL word of a
        # multi-word name (the German noun head: "Feuerfalter" in
        # "Großer Feuerfalter") from its adjective lead words: a hit
        # on the noun head is strong (rank 1), a hit only on a lead
        # adjective ("groß" in "Großer ...") is weak (rank 2) - the
        # user asked about an aspect, not that species (design D9).
        ranked = []  # (rank, name)
        for token in input_stems:
            loose_ok = len(token) >= 4 and token not in botText.FUNCTION_WORD_STEMS
            for name, full_stem, word_stems, raw_words in item_stems:
                if full_stem == token:
                    ranked.append((0, name))
                elif token in word_stems:
                    if len(word_stems) == 1 or word_stems[-1] == token:
                        ranked.append((1, name))
                    else:
                        ranked.append((2, name))
                elif loose_ok and any(
                        ws.startswith(token) or ws.endswith(token)
                        for ws in list(word_stems) + list(raw_words)
                        if ws != token):
                    # Distinguish the German compound noun head from lead
                    # adjectives: a token that TERMINATES the final word
                    # ("frosch" in "Teichfrosch") names the species - rank
                    # 1. A token that only prefixes a lead word ("groß" in
                    # "Großer Feuerfalter") is aspect noise - rank 2 (weak).
                    head_words = ([raw_words[-1]] if raw_words else []) + \
                                 ([word_stems[-1]] if word_stems else [])
                    if any(ws.endswith(token) for ws in head_words if ws != token):
                        ranked.append((1, name))
                    else:
                        ranked.append((2, name))

        if ranked:
            # Deduplicate, order by best rank, stable within a rank.
            # A result resting ONLY on rank-2 loose hits is marked weak
            # (design D9): such a "match" is prefix noise (e.g. "groß"
            # vs. all Groß-* entities), not a real entity mention, and
            # the caller may defer it to the focus follow-up.
            seen = set()
            ordered = []
            min_rank = None
            for rank, name in sorted(ranked, key=lambda r: r[0]):
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
            # Group cap (limit-group-options-to-five): offer at most five
            # species. When the ranks distinguish the members, keep the
            # best-ranked five in rank order; when all share one rank
            # (no ranking signal), pick five at random so repeated asks
            # can surface different members. The weak flag is computed
            # from min_rank BEFORE the cap.
            if len(ordered) > 5:
                if min_rank != max(r for r, _ in ranked):
                    ordered = ordered[:5]
                else:
                    ordered = random.sample(ordered, 5)
            if self.DEBUG:
                print(f"Stage 1 (stem containment) matched: {ordered}")
            weak = (min_rank == 2)
            if len(ordered) == 1:
                return {"kind": "single", "name": ordered[0], "weak": weak}
            return {"kind": "group", "names": ordered, "weak": weak}

        # Stage 2: per-token fuzzy fallback against item names.
        # Short token stems (function words like "wie", "wett") get
        # inflated WRatio scores against long names and would match
        # spuriously, so the fallback only considers stems of 5+
        # characters - long enough for real typo-style queries.
        #
        # A candidate is only accepted when the token stem and the
        # candidate name stem share their first three characters
        # (umlaut-mapped on both sides): WRatio inflates scores for
        # lookalikes ("feuchtwies" -> "efeu" at 77.1, "luchs" ->
        # "rotfuchs" at 80.0), while a genuine typo shares the prefix
        # ("feuchtwies" -> "feuchtwiesen" at 90.9). Differently-prefixed
        # typos are left to the entity arbiter (fix-entity-resolution-
        # fallback design D2/D1).
        best = None  # (score, name)
        for token in botText.normalize(input_text):
            if len(token) < 5:
                continue
            matches = process.extract(
                token, [str(i).lower() for i in items],
                scorer=fuzz.WRatio, processor=matchProcessor, limit=5,
            )
            for cand, score, idx in matches:
                if score < self.matchThreshold:
                    break  # matches are ordered by score descending
                if german_stem(str(items[idx]))[:3] != token[:3]:
                    continue
                if best is None or score > best[0]:
                    best = (score, str(items[idx]))
                break  # best prefix-compatible candidate for this token

        if best and best[0] >= self.matchThreshold:
            if self.DEBUG:
                print(f"Stage 2 (fuzzy fallback) matched: {best[1]} (score {best[0]})")
            return {"kind": "single", "name": best[1]}

        if self.DEBUG:
            print("No staged match found.")
        return None

    # ------------------------------------------------------------------
    # Entity arbiter (fix-entity-resolution-fallback): LLM-assisted
    # recovery on the no-match path. See the entity-arbiter spec.
    # ------------------------------------------------------------------
    ARBITER_SYSTEM_PROMPT = (
        "Du bist ein Entitäts-Klassifizierungssystem für einen Chatbot über "
        "Tiere, Pflanzen und Lebensräume (Auen) im Karlsruher Rheinauen-Gebiet. "
        "Dir wird eine Nutzereingabe gegeben sowie drei Listen mit gültigen "
        "Entitätsnamen, gruppiert nach Domäne (Tier, Pflanze, Auen). "
        "Bestimme, welche Entität aus den Listen die Nutzereingabe meint - "
        "auch wenn der Name nur Teil eines längeren Satzes ist oder leicht "
        "falsch geschrieben ist. Antworte NUR mit JSON im Format "
        '{"found": true/false, "domain": "Tier"|"Pflanze"|"Auen", "name": "<Name aus den Listen>"} . '
        "Wenn die Eingabe keine Entität aus den Listen meint, antworte mit "
        '{"found": false, "domain": null, "name": null}. '
        "Der Name MUSS exakt einem Namen aus den Listen entsprechen."
    )

    def _get_llm(self):
        """Return the injected LLM client, or lazily build one from the
        private config (same fallback pattern as botCtl.py)."""
        if self.llm is not None:
            return self.llm
        try:
            import private as pr
            llm = OpenAICompatClient(
                base_url=pr.baseUrl,
                api_key=pr.apiKey,
                emb_url=getattr(pr, "embUrl", pr.baseUrl),
                chat_model=pr.lngMdl,
                embed_model=getattr(pr, "embMdl", None),
            )
            llm.setDebug(self.DEBUG)
            self.llm = llm
            return llm
        except Exception as e:
            if self.DEBUG:
                print("Arbiter: no LLM client available:", e)
            return None

    def _domain_lists(self):
        """Load and cache the arbiter candidate lists from the three
        domain files. Returns {"Tier": [...], "Pflanze": [...], "Auen": [...]}
        or None when a file is missing."""
        if self._domain_lists_cache is not None:
            return self._domain_lists_cache
        lists = {}
        try:
            for domain, fname in (("Tier", "entity_tiere.json"),
                                  ("Pflanze", "entity_pflanzen.json"),
                                  ("Auen", "entity_aue.json")):
                path = os.path.join(os.path.dirname(__file__), "data", fname)
                with open(path, "r", encoding="utf-8") as f:
                    lists[domain] = [str(n).strip() for n in json.load(f)]
        except Exception as e:
            if self.DEBUG:
                print("Arbiter: candidate lists unavailable:", e)
            return None
        self._domain_lists_cache = lists
        return lists

    def followup_check(self, user_text, focus_entity):
        """LLM follow-up gate for the focus fill (conversation-recovery).

        Asks once whether the user input is a follow-up question about
        the focused entity. Returns True only on a clear yes; False on
        a clear no, a malformed answer, or an LLM failure (the caller
        then stays on the no-match path, so a broken LLM degrades to
        the pre-follow-up behavior and the breakout stays reachable).
        """
        llm = self._get_llm()
        if llm is None:
            return False
        try:
            answer = llm.chat_json(
                temperature=0.0,
                system=(
                    "Du bist ein Dialog-Klassifizierer für einen Natur-Chatbot. "
                    "Ein Nutzer hat zuvor eine Frage über eine Entität gestellt "
                    "und sie wurde beantwortet. Entscheide, ob die neue "
                    "Nutzereingabe eine Follow-up-Frage ÜBER diese Entität ist "
                    "(z.B. nach einem anderen Aspekt derselben Entität fragen, "
                    "oder mit Pronomen darauf verweisen). Antworte NUR mit JSON: "
                    '{"followup": true/false}. Eine Eingabe, die die Entität '
                    "gar nicht meint oder unsinnig ist, ist KEIN Follow-up."
                ),
                user=(
                    f"Vorherige Entität: '{focus_entity}'.\n"
                    f"Neue Nutzereingabe: '{user_text}'"
                ),
                schema_hint='{"followup": bool}',
            )
        except Exception as e:
            if self.DEBUG:
                print("Follow-up check: LLM call failed:", e)
            return False
        if self.DEBUG:
            print("Follow-up check: raw answer:", answer)
        return isinstance(answer, dict) and answer.get("followup") is True

    def _validate_arbiter_answer(self, answer):
        """Strictly validate an arbiter answer against the domain lists.

        Returns the validated entity name, or None. An answer is valid
        only when `found` is true, the domain is one of the three
        candidate domains, and the name matches exactly one entry of
        THAT domain's file (case-insensitive, whitespace-trimmed).
        """
        if not isinstance(answer, dict):
            return None
        if not answer.get("found"):
            return None
        domain = answer.get("domain")
        name = answer.get("name")
        if not isinstance(domain, str) or not isinstance(name, str):
            return None
        lists = self._domain_lists()
        if lists is None or domain not in lists:
            return None
        wanted = name.strip().lower()
        for candidate in lists[domain]:
            if candidate.lower() == wanted:
                return candidate
        return None

    def arbiter_resolve_entity(self, input_text):
        """LLM-assisted entity recovery for the no-match path.

        Makes exactly one chat call classifying the input against the
        three domain candidate lists. Returns the validated entity name
        (canonical casing from the candidate file) or None - None covers
        "not found", invalid/hallucinated answers, and LLM failures; the
        caller continues on the existing no-match path in all of them.
        """
        lists = self._domain_lists()
        llm = self._get_llm()
        if lists is None or llm is None:
            return None

        lists_text = "\n".join(
            f"{domain} ({len(names)} Namen): {', '.join(names)}"
            for domain, names in lists.items()
        )
        try:
            answer = llm.chat_json(
                temperature=0.0,
                system=self.ARBITER_SYSTEM_PROMPT,
                user=f"Nutzereingabe: '{input_text}'\n\n{lists_text}",
                schema_hint='{"found": bool, "domain": "Tier"|"Pflanze"|"Auen"|null, "name": string|null}',
            )
        except Exception as e:
            if self.DEBUG:
                print("Arbiter: LLM call failed:", e)
            return None

        if self.DEBUG:
            print("Arbiter: raw answer:", answer)
        validated = self._validate_arbiter_answer(answer)
        if self.DEBUG:
            print("Arbiter: validated name:", validated)
        return validated

    def _match_input_to_options(self, input_text, options):
        """
        Match user input against a list of option dicts.
        
        Args:
            input_text: User input string
            options: List of option dicts with "title" key
            
        Returns:
            Matched option title or None if no match
        """
        if self.DEBUG:
            print(f"_match_input_to_options: Matching input '{input_text}' against options")
        option_titles = [o.get("title", "").lower() for o in options]
        if self.DEBUG:
            print("Options: ", option_titles)

        return self._match_input_to_items(input_text, option_titles)

    def _process_action(self, intent, input_text, context={}, lang="de"):
        """
        Process the action defined in an intent.
        
        Args:
            intent: Intent dict
            input_text: User input string
            lang: Language code
            
        Returns:
            Action result dict or empty dict if no action
        """
        if self.DEBUG:
            print(f"_process_action: Processing action for intent '{intent.get('intent')}'")
        action = intent.get("action", None)
        if self.DEBUG:
            print("Action to process:", action)
        
        if action is not None and action.strip() != "":
            # check if action is defined in actionCtl
            if self.actionCtl is not None and action in self.actionNames:
                if self.DEBUG:
                    print(f"Executing action '{action}' via BotActions.")
                input = input_text or ""
                status, action_result = self.actionCtl.execute(action, input=input, context=context, lang=lang)
                if self.DEBUG:
                    print("Action result:", status, action_result)
                return action_result
            # fallback: use dummy action
            else:
                if self.DEBUG:
                    print(f"Executing dummy action '{action}'.")            
                input = input_text or ""
                action_result = BotIntent.dummyAction(action, input=",".join([input,json.dumps(list(context.keys()))]), lang=lang)
                if self.DEBUG:
                    print("Action result:", action_result)
                return action_result


    def _update_context_with_provides(self, intent, context):
        """
        Update context with values from intent's "provides" field.
        
        Args:
            intent: Intent dict
            context: Context dict to update (modified in-place)
        """
        if self.DEBUG:
            print(f"_update_context_with_provides: Updating context with provides from intent '{intent.get('intent')}'")
        provides = intent.get("provides") or {}
        if self.DEBUG:
            print("Provided:", provides)
        for key, value in provides.items():
            if self.DEBUG:
                print(f"Updating context with provided '{key}': '{value}'")
            context[key] = value

    def _finalize_completion(self, intent, context, requirements, input_text, lang):
        """
        Finalize a completed intent by processing action and cleaning up context.
        
        Args:
            intent: Intent dict
            context: Current context dict
            requirements: List of requirement names to remove from context
            input_text: User input string
            lang: Language code
            
        Returns:
            dict with "output" and "context" keys
        """
        if self.DEBUG:
            print(f"_finalize_completion: Finalizing completion for intent '{intent.get('intent')}'")

        ctx = dict(context)
        ctx.pop("options", None)
        # insert input as last_input in context
        # context["last_input"] = input_text

        # Conversation trace + focus (add-no-match-breakout-and-focus-
        # followups, design D4/D5): every completed answer records what
        # was answered and makes the resolved entity the conversation
        # focus for follow-up questions.
        ctx["last_input"] = input_text
        ctx["last_intent"] = intent.get("intent")
        if context.get("entity"):
            ctx["focus"] = {"entity": context.get("entity"), "type": context.get("type")}

        # Check for target routing
        if "target" in requirements and "target" in context:
            target_name = context.get("target")
            matched_intent = self.get_intent_by_name(target_name)
            if matched_intent is None:
                if self.DEBUG:
                    print(f"Target intent '{target_name}' not found.")
                return self.__make_error_response("intent_not_found", ctx, intent, lang)

            if self.DEBUG:
                print(f"Routing to matched intent '{target_name}' -> {matched_intent}")
            
            output = {"text": f"{BotIntent.getMsg('completed', lang)} Route to intent: {matched_intent.get('intent') if matched_intent else target_name}"}
            # overwrite intent with target (D1: routed flow starts fresh;
            # its own completion pops the marker again)
            ctx = { "intent": target_name, "last_input": input_text }
            if self.DEBUG:
                print(f"Executing routed intent '{target_name}'")
            routed = self.execute(intent_name=target_name, context=ctx, input=input_text, lang=lang)
            # The routed flow answered in this same call (e.g. a default
            # handler): its marker must not survive either, or the sticky
            # routing starts over from the routed intent. A still-pending
            # routed flow keeps its marker (it is waiting for input).
            if routed.get("completed"):
                routed_ctx = dict(routed.get("context") or {})
                routed_ctx.pop("intent", None)
                routed["context"] = routed_ctx
            return routed

        else:
            # Process action or use default completion message
            action_result = self._process_action(intent, input_text, ctx, lang)
            output = action_result if action_result else {"text": BotIntent.getMsg("completed", lang)}

            # Feature fallback (add-no-match-breakout-and-focus-followups,
            # design D8): when the requested feature has no data for the
            # resolved entity, inform the user and offer up to 3 of the
            # entity's data-carrying features as options. The entity stays
            # in the context, `feature` becomes the missing requirement, and
            # the existing options flow completes the follow-up pick.
            if action_result and action_result.get("no_feature_data"):
                available = action_result.get("available_features") or []
                if available:
                    ctx.pop("feature", None)
                    ctx["options"] = [{"title": f} for f in available]
                    feat = context.get("feature") or "dieser Eigenschaft"
                    output = {"text": (
                        f"Zur Eigenschaft '{feat}' habe ich leider keine Informationen. "
                        "Über einen dieser Aspekte kann ich dir etwas erzählen:"
                    )}
                    # marker for the miss counter (design D10): the turn
                    # could not answer - if it was a focus fill, it counts.
                    return {"output": output, "context": ctx, "no_feature_data": True}

        # Remove requirements from context
        for req in requirements:
            ctx.pop(req, None)

        # Pending-completion marker (reset-sticky-intent-after-completion,
        # D1): `intent` is only set while a flow waits for user input.
        # The flow ends here, so the marker must go, or the router keeps
        # forcing every later turn into this intent. The feature-fallback
        # early return above is deliberately NOT affected: that flow is
        # still waiting for the feature pick. last_intent / last_input /
        # focus are the durable memory fields and stay.
        ctx.pop("intent", None)

        return {"output": output, "context": ctx,"completed": True}

    def __defaultHandler(self, intent, input=None, context=None, lang="de"):
        """Simple handler that returns the configured utterance and optionally
        processes a placeholder action. Does not perform completion or matching."""
        if self.DEBUG:
            print("Default handler for intent:", intent.get("intent"), "lang:", lang)
            print("Input:", input)
            print("Context (in):", context)

        context = context or {}

        locale = "" if lang is None else f"_{lang}"
        text = intent.get(f"utter{locale}", "") or ""
        output = {"text": text}

        if intent.get("link") is not None:
            output["link"] = intent.get("link")

        # Process action if present
        action_result = self._process_action(intent, input, context, lang)
        if action_result:
            for key, value in action_result.items():
                if key != "text":
                    output[key] = value
                else:
                    output[key] = (output[key] + " " + value).strip()

        # insert input as last_input in context
        # context["last_input"] = input
        # Pending-completion marker (D1): the default handler answers
        # immediately, so an intent marker the router attached must not
        # survive this turn.
        context.pop("intent", None)
        return {"output": output, "context": context,"completed": True}

    def __completeHandler(self, intent, input=None, context=None, lang="de"):
        """
        Handler for completion-style intents. Manages multi-step interactions
        by presenting options or matching input against required items.
        """
        if self.DEBUG:
            print("Complete handler for intent:", intent.get("intent"), "lang:", lang)
            print("Input:", input)
            print("Context (in):", context)

        context = context or {}
        intent_base = (intent.get("intent", "").split("_")[0] or "").strip()
        
        if self.DEBUG:
            print("Intent base:", intent_base)

        # Update context with provided values
        self._update_context_with_provides(intent, context)

        # Parse requirements
        reqs = [r.strip() for r in (intent.get("requires") or "").split(",") if r.strip() != ""]
        if self.DEBUG:
            print("Intent requires:", reqs)

        # Handle options presentation or input matching
        ctx_opts = context.get("options")
        # insert input as last_input in context
        # context["last_input"] = input
        
        if ctx_opts is None:
            return self._handle_no_options(intent, input, context, reqs, intent_base, lang)
        else:
            return self._handle_with_options(intent, input, context, ctx_opts, reqs, lang)

    def _handle_no_options(self, intent, input_text, context, requirements, intent_base, lang):
        """
        Handle completion when no options are currently in context.
        Either present options or match input against items.
        """
        if self.DEBUG:
            print(f"_handle_no_options: No options in context")
            print("Intent:", intent)
        # Present options if configured
        option_name = (intent.get("options") or "").strip()
        if self.DEBUG:
            print("Option name from intent:", option_name)
        if option_name:
            if self.DEBUG:
                print("Presenting options for completion:", option_name)
            
            options = self._load_options_file(option_name)
            if options is None:
                return self._make_error_response("options_file_not_found", context, intent, lang)

            ctx = dict(context)
            ctx["options"] = options
            # Pending-completion marker (D1): set only while this flow
            # waits for user input; popped again on completion.
            ctx["intent"] = intent.get("intent")
            
            output = {"text": BotIntent.getMsg("select_options", lang)}
            if intent.get("link") is not None:
                output["link"] = intent.get("link")
            
            return {"output": output, "context": ctx}
            # ########## completed #############

        # No options configured - match input against items
        if not input_text or not input_text.strip():
            return self._make_error_response("no_input_completion", context, intent, lang)

        # Find first missing requirement and its items
        missing_req = self._find_first_missing_requirement(requirements, context)
        if self.DEBUG:
            print("First missing requirement:", missing_req)
            
        if missing_req is None:
            if self.DEBUG:
                print("All requirements satisfied.")
            return self._finalize_completion(intent, context, requirements, input_text, lang)
            # ########## completed #############
        

        items = self._load_items_file(missing_req, intent_base)
        if items is None:
            return self._make_error_response("no_items_file", context, intent, lang)

        # Match input to items (staged matcher, design D2). A group
        # match offers the members as options instead of filling the
        # requirement arbitrarily; a single match fills it directly.
        match_result = self._match_input_to_items_structured(input_text, items)
        arbiter_used = False
        focus_filled = False
        focus = context.get("focus") if isinstance(context.get("focus"), dict) else {}
        focus_entity = focus.get("entity") if missing_req == "entity" else None

        if match_result is None:
            if self.DEBUG:
                print("No staged match found for input.")
            # Entity arbiter (fix-entity-resolution-fallback): one LLM
            # recovery attempt before giving up.
            recovered = self.arbiter_resolve_entity(input_text) if missing_req == "entity" else None
            if recovered is not None:
                if self.DEBUG:
                    print(f"Arbiter recovered '{missing_req}': '{recovered}'")
                arbiter_used = True
                match_result = {"kind": "single", "name": recovered}
                # The recovered entity usually belongs to a different domain
                # than the decoded intent (e.g. an Auen name under a tp_*
                # intent), so the intent-provided feature often has no data
                # for it (e.g. 'generell'). 'definition' maps to the full
                # key list (design D7), which every entity has content for.
                context["feature"] = "definition"
            elif focus_entity:
                # Focus-based follow-up (add-no-match-breakout-and-focus-
                # followups, design D3/D10): the LLM follow-up gate decides
                # whether the input refers to the previously answered
                # entity. The intent's feature is KEPT (the follow-up names
                # a concrete aspect). A "no" (or LLM failure) stays on the
                # no-match path, so the breakout stays reachable.
                if self.followup_check(input_text, focus_entity):
                    if self.DEBUG:
                        print(f"Focus follow-up confirmed: filling '{missing_req}' from focus: '{focus_entity}'")
                    focus_filled = True
                    match_result = {"kind": "single", "name": focus_entity}
        if match_result is None:
            # User-appropriate no-match text (D4); the internal marker
            # "no_match_completion" stays on the result for history.
            result = self._make_error_response("no_match_completion", context, intent, lang)
            result["no_match"] = True
            return result

        if match_result["kind"] == "group" and missing_req == "entity":
            # A weak group (only adjective/loose hits, design D9) defers
            # to the focus when one exists: e.g. "und wie groß ist sie?"
            # weakly matches every Groß-* entity, but the user means the
            # entity of the previous answer - the LLM follow-up gate
            # decides (same gate as the unmatched path above).
            focus_entity = focus.get("entity") if isinstance(focus, dict) else None
            if match_result.get("weak") and focus_entity and self.followup_check(input_text, focus_entity):
                if self.DEBUG:
                    print(f"Weak group defers to confirmed focus: '{focus_entity}'")
                focus_filled = True
                match_result = {"kind": "single", "name": focus_entity}
            else:
                names = match_result["names"]
                if self.DEBUG:
                    print(f"Group match for '{missing_req}', offering options: {names}")
                ctx = dict(context)
                ctx["options"] = [{"title": name} for name in names]
                ctx["intent"] = intent.get("intent")
                output = {"text": BotIntent.getMsg("select_options", lang)}
                if intent.get("link") is not None:
                    output["link"] = intent.get("link")
                return {"output": output, "context": ctx}

        # Single match (or group collapsed to its best member for
        # non-entity requirements) fills the requirement as today.
        if match_result["kind"] == "group":
            matched_value = match_result["names"][0]
            if self.DEBUG:
                print(f"Group match for non-entity requirement '{missing_req}', using best member: {matched_value}")
        else:
            matched_value = match_result["name"]

        # Update context and finalize
        context[missing_req] = matched_value
        if self.DEBUG:
            print(f"Updating context with '{missing_req}': '{matched_value}'")

        # Check if all requirements satisfied
        all_satisfied = all(req in context for req in requirements)
        if self.DEBUG:
            print("Requirements satisfied?:", all_satisfied, "Requirements:", requirements, "Context keys:", list(context.keys()))
        
        if all_satisfied:
            if self.DEBUG:
                print("All requirements satisfied.")
            result = self._finalize_completion(intent, context, requirements, input_text, lang)
            if arbiter_used:
                result["llm_used"] = True
            if focus_filled:
                result["focus_fill"] = True
            return result
        else:
            if self.DEBUG:
                print("Some requirements still missing.")
            # Not completed - continue without options
            ctx = dict(context)
            ctx["intent"] = intent.get("intent")
            
            out = {"output": {"text": BotIntent.getMsg("continue_completion", lang)}, "context": ctx}
            if arbiter_used:
                out["llm_used"] = True
            if focus_filled:
                out["focus_fill"] = True
            return out

    def _handle_with_options(self, intent, input_text, context, options, requirements, lang):
        """
        Handle completion when options are present in context.
        Match selected option and update requirements.
        """
        if self.DEBUG:
            print(f"_handle_with_options: Options present in context")
        if not input_text or not input_text.strip():
            return self._make_error_response("no_input_completion", context, intent, lang)

        # Match input to options
        matched = self._match_input_to_options(input_text, options)
        if matched:
            if self.DEBUG:
                print("Matched")
            # Fill first missing requirement
            missing_req = self._find_first_missing_requirement(requirements, context)
            if missing_req:
                context[missing_req] = matched
                if self.DEBUG:
                    print(f"Updating context with '{missing_req}': '{matched}'")
            elif "feature" in requirements and matched:
                # Feature-pick continuation (conversation-recovery D8): the
                # offered options were data-carrying features of an already
                # resolved entity; the intent's provides re-added the empty
                # feature, so override it with the user's pick.
                context["feature"] = matched
                if self.DEBUG:
                    print(f"Feature pick: overriding feature with '{matched}'")
        else:
            if self.DEBUG:
                print("Not matched")
            result = self._make_error_response("no_match_completion", context, intent, lang)
            result["no_match"] = True
            return result
        
        # Check if all requirements satisfied
        all_satisfied = all(req in context for req in requirements)
        if self.DEBUG:
            print("Requirements satisfied?:", all_satisfied, "Requirements:", requirements, "Context keys:", list(context.keys()))
        
        if all_satisfied:
            if self.DEBUG:
                print("All requirements satisfied.")
            return self._finalize_completion(intent, context, requirements, input_text, lang)
        else:
            if self.DEBUG:
                print("Some requirements still missing.")
            # Not completed - continue with options
            ctx = dict(context)
            ctx["options"] = options
            ctx["intent"] = intent.get("intent")
            
            return {"output": {"text": BotIntent.getMsg("continue_completion", lang)}, "context": ctx}
    



if __name__ == "__main__":
    import sys
    intent = BotIntent("../rawData/intents.json")
    #intent = BotIntent("./data/test_intents.json")
    intent.setActions("../rawData/tiere_pflanzen_auen.json")
    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        intent.setDebug(True)
    lang = "de"
    print(f"Loaded intent '{intent.name}' with {len(intent.data)} entries.")
    for i in intent.data:
        print(i["intent"],i["id"])
        result = intent.execute(i["intent"], context={"intent": i["intent"]}, lang=lang)
        print("Result:", result)
        ctx = result.get("context",{})
        if "options" in ctx:
            print("  -> Options:", [o["title"] for o in ctx["options"]])
            print("\nSimulate selection of an option...")
            result = intent.execute(i["intent"], context=result["context"], input=random.choice(ctx["options"])["title"], lang=lang)
            print("Result2:", result)
        else:
            ireqs = intent.get_intent_by_name(i["intent"]).get("requires","") or ""
            ireqs = [r.strip() for r in ireqs.split(",") if r.strip() != ""]
            reqs = [r for r in ctx.keys() if r in ireqs]
            for r in ireqs:
                print(f"  -> Requirement '{r}'")
                if r not in reqs:
                    print("\nSimulate input for requirement...")
                    # load items file
                    items = intent._load_items_file(r, i["intent"].split("_")[0])
                    if items:
                        simulated_input = random.choice(items)
                        print(f"Simulated input: {simulated_input}")
                        result = intent.execute(i["intent"], context=result["context"], input=simulated_input, lang=lang)
                        print("Result2:", result)
                    else:
                        print(f"No items file for requirement '{r}' and intent base '{i['intent'].split('_')[0]}'")
                        result = intent.execute(i["intent"], context=result["context"], input="abxx", lang=lang)
                        print("Result2:", result)

        print("\n\n")


    exit()
        
    # Intent matching test (expertise)
    print("\n--- Intent matching Test ---")
    test_intent = "test8"

    result = intent.execute(test_intent, input=None, context={"intent": test_intent}, lang=lang)
    print("Result with no input:", result)        
    print("\n\n")
    result = intent.execute(test_intent, input="test1", context=result["context"], lang=lang)
    print("Result with input 'test1':", result)        
    print("\n\n")
    result = intent.execute(test_intent, input="xxx", context={"intent": test_intent}, lang=lang)
    print("Result with input 'xxx':", result)
    print("\n\n")
    
    print("\n--- Feature matching Test ---")
    test_intent = "test2"

    result = intent.execute( test_intent, input=None, context={"intent": test_intent}, lang=lang)
    print("Result with no input:", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="Feat2_4", context=result["context"], lang=lang)
    print("Result with input 'Feat2_4':", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="feat3", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feat3':", result)
    print("\n\n")
    result = intent.execute( test_intent, input="feaxx", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feaxx':", result)
    print("\n\n")
    
    print("\n--- Feature matching Test with ent,type ---")
    test_intent = "test2b"
    test_handler = intent.get_intent_by_name(test_intent)["handler"]

    result = intent.execute( test_intent, input=None, context={"intent": test_intent}, lang=lang)
    print("Result with no input:", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="Feat2_4", context=result["context"], lang=lang)
    print("Result with input 'Feat2_4':", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="feat3", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feat3':", result)
    print("\n\n")
    result = intent.execute( test_intent, input="feaxx", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feaxx':", result)
    print("\n\n")
    
    