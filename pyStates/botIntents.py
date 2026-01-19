import json
import os

class BotIntent:
    def __init__(self, path, parameters=None):
        self.name = path
        self.name = os.path.basename(path)
        self.parameters = parameters or {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            # Expecting a list of dicts. If it's a dict, try to convert to list.
            if isinstance(loaded, dict):
                # common case: dict with numeric keys or single record
                if all(isinstance(v, dict) for v in loaded.values()):
                    self.data = list(loaded.values())
                else:
                    # fallback: wrap into single-element list
                    self.data = [loaded]
            elif isinstance(loaded, list):
                self.data = loaded
            else:
                self.data = []

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{self.name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}")

    def get_intent_by_id(self, intent_id, lang="de"):
        for entry in self.data:
            if entry.get("id") == intent_id:
                key = f"intent_{lang}"
                name = entry.get(key,None)
                output = entry.get(f"utter_{lang}", None)
                alias = entry.get(f"alias_{lang}", None)
                return {"name":name, "output": output, "alias": alias}
        return None

if __name__ == "__main__":
    intent = BotIntent("../rawData/intents_translated.json")
    print(f"Loaded intent '{intent.name}' with {len(intent.data)} entries.")
    for i in intent.data[:5]:
        print(i["intent_de"],i["id"])
    