from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from bot_router import build_router


def main() -> None:
    intents_path = "../rawData/intents.json"
    context_path = "../rawData/tiere_pflanzen_auen.json"
    # read intents here already
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_ = json.load(f)

    intents = {intent["id"]: intent for intent in intents_}
        
    try:
        # import private as pr  # type: ignore
        private = {
            "apiKey": "1234",
            "baseUrl": "http://localhost:8080",
            "embUrl": "http://localhost:8085",
            "lngMdl": "ibm-granite.granite-4.0-h-1b.Q4_K_M",
            "embMdl": "bge-m3",
        }
        print("Loaded private config for LLM.")
    except Exception:
        print("No private config found for LLM.")
        private = None

    router = build_router(intents_path, context_path, llm_threshold=0.15, private=private,ranking = None) #["bm25"])

    # iterate through intents. for each text item in each intent, vectorize text
    
    vecs = []
    ints = []
    texts_list = []
    text_len = []

    for intent in intents:
        print(f"Checking intent {intent}...",intents[intent]["intent"])
        # try common keys for example texts
        texts = intents[intent]["text"] if "text" in intents[intent] else None
        if not texts:
            continue

        embeddings = router.vectorize(texts)
        for emb in embeddings:
            # ensure pure Python floats for JSON serialization
            vecs.append([float(x) for x in emb])
            ints.append(intent)
            texts_list.append(texts)
            text_len.append([len(t.split(" ")) for t in texts])


    out_path = "intent_vectors.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"vectors": vecs, "intents": ints, "texts": texts_list, "text_lengths": text_len}, f, ensure_ascii=False, indent=2)
    print(f"Wrote intent vectors to {out_path}")
    
    


if __name__ == "__main__":
    main()
