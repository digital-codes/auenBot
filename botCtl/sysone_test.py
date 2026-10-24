"""
sysone_test.py - smoke test for SystemOne routing.

Runs the stage-1 router and the stage-2 category decision against the live
endpoint and checks the local category answer building. Run from botCtl/:

    python3 sysone_test.py
"""
import json

from botSystemone import SystemoneRouter, INTENT_ROUTING
from botActions import BotAction

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


router = SystemoneRouter()
router.setDebug(False)
print(f"SystemOne routing enabled: {router.enabled()}")

if not router.enabled():
    raise SystemExit("SystemOne not enabled (missing SYSONE_KEY/SYSONE_URL or data files)")

# ------------------------------------------------------------------ stage 1
print("\n--- Stage 1 (intent + transitions) ---")
cases = [
    # (label, query, focus, expected intent, cond)
    ("bio_nahrung", "Was frisst der Eisvogel?", None, "bio_frage", {}),
    ("bio_lat", "Was ist Malus sylvestris?", None, "bio_frage", {}),
    ("smalltalk", "Hallo!", None, "smalltalk", {}),
    ("unrelated", "Wie wird das Wasserpreise morgen in Karlsruhe?", None, None,
     {"unrelated_at_least": 0.8}),
    ("nazka", "Wann hat das Naturschutzzentrum offen?", None, "nazka_info", {}),
    ("followup", "Und wie schwer wird er?", "Eisvogel", "bio_frage",
     {"followup_at_least": 0.7}),
    ("topic_switch", "Wo lebt der Moorfrosch?", "Eisvogel", "bio_frage",
     {"new_topic_at_least": 0.7}),
]
for label, query, focus, want, conds in cases:
    try:
        res = router.detect(query, focus_entity=focus, prev_answered=bool(focus))
        ok = True
        detail = []
        if want is not None and res["intent"] != want:
            ok = False
            detail.append(f"intent={res['intent']} want={want}")
        for k, v in conds.items():
            if k.endswith("_at_least"):
                got = res[k.replace("_at_least", "")]
                if got < v:
                    ok = False
                    detail.append(f"{k}={got:.2f}<{v}")
        check(label, ok, " ".join(detail) or f"intent={res['intent']}")
    except Exception as ex:
        check(label, False, f"error: {ex}")

print("\n  Routing map sample:", INTENT_ROUTING["bio_frage"],
      INTENT_ROUTING["smalltalk"], INTENT_ROUTING["essen"])

# ------------------------------------------------------------------ stage 2
print("\n--- Stage 2 (category) ---")
for label, query, entity, want_cat in [
    ("kat_verhalten", "Was frisst der Eisvogel?", "Eisvogel", "verhalten"),
    ("kat_eigenschaften", "Wie groß wird der Eisvogel?", "Eisvogel", "eigenschaften"),
    ("kat_beschreibung", "Was ist ein Moorfrosch?", "Moorfrosch", "beschreibung"),
]:
    try:
        cat = router.category(query, entity)
        check(label, cat["category"] == want_cat and not cat["tie"],
              f"got={cat['category']} tie={cat['tie']} conf={cat['confidence']}")
    except Exception as ex:
        check(label, False, f"error: {ex}")

# ------------------------------------------------- local answer building
print("\n--- Answer building (no endpoint) ---")
action = BotAction("../rawData/tiere_pflanzen_auen.json")
action.sysone = SystemoneRouter()
action.sysone._enabled = False  # force the plain category path
action.setDebug(False)

feat = action.get_entity_features("Eisvogel", "verhalten")
check("verhalten_columns", len(feat.get("text", [])) >= 3,
      f"n_texts={len(feat.get('text', []))}")
check("verhalten_audio", len(feat.get("audio", [])) > 0, str(feat.get("audio"))[:80])

feat = action.get_entity_features("Moorfrosch", "beschreibung")
check("beschreibung_columns", any("Rana arvalis" in str(t) or "Rana" in str(t) for t in feat.get("text", [])),
      f"text={str(feat.get('text'))[:100]}")
check("beschreibung_image", len(feat.get("image", [])) > 0, str(feat.get("image"))[:80])

feat = action.get_entity_features("Stieleiche", "eigenschaften")
check("eigenschaften_filled", len(feat.get("text", [])) > 0, str(feat.get("text"))[:80])

filled = action._filled_features("Eisvogel", limit=3)
check("filled_features", 0 < len(filled) <= 3, str(filled))

# ------------------------------------------------- full bio_feature action
print("\n--- handle_bio_feature (with SystemOne) ---")
action.sysone = router
ok, out = action.handle_bio_feature(input="Was frisst der Eisvogel?",
                                    context={"entity": "Eisvogel", "feature": ""})
check("bio_action_verhalten", ok and isinstance(out.get("text"), list) and len(out["text"]) > 0,
      str(out)[:120])
check("bio_action_audio", len(out.get("audio", [])) > 0, str(out.get("audio"))[:80])

# tie path: force tie by lowering thresholds on a copy
import botSystemone
old_margin, old_conf = botSystemone.TIE_MARGIN, botSystemone.TIE_CONFIDENCE
botSystemone.TIE_MARGIN, botSystemone.TIE_CONFIDENCE = 1.0, 1.0
ok, out = action.handle_bio_feature(input="Was gibt es über die Gelbbauchunke?",
                                    context={"entity": "Gelbbauchunke", "feature": ""})
botSystemone.TIE_MARGIN, botSystemone.TIE_CONFIDENCE = old_margin, old_conf
check("bio_action_tie_overview", ok and any("interessiert dich" in t for t in out.get("text", [])),
      str(out.get("text"))[:100])
check("bio_action_tie_options", len(out.get("options", [])) >= 2, str(out.get("options")))

print(f"\n=== {PASS} PASS / {FAIL} FAIL")
raise SystemExit(1 if FAIL else 0)
