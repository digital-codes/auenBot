# AuenBot
Migration from LUBW/NAZKA ChatBot KarlA


# Deploy

All runtime parts live in **botCtl** (Flask app, port 11354). Deployment steps:

1. **Checkout with git lfs** - `rawData/intent_vectors.json` (~119 MB embeddings) and other media files are LFS-tracked. A plain clone leaves pointer files behind and the legacy decoder cannot load.
2. **Python dependencies** - flask, jsonschema, sqlalchemy, numpy, requests, rapidfuzz (or fuzzywuzzy), pandas (only for `makeDataFiles.py`), cryptography.
3. **Config: `botCtl/private.py`** (not in the repo) - LLM/embedding access (`apiKey`, `baseUrl`, `embUrl`, `embMdl`, `lngMdl`) and, for SystemOne routing, `SYSONE_KEY`, `SYSONE_URL` (default `https://api.greenpt.ai/v1/systemone`), `SYSONE_MDL` (`metis`) plus the optional `SYSONE_ROUTING` switch (default: on when `SYSONE_KEY` exists or a non-default `SYSONE_URL` is set, e.g. a local Ollama SystemOne endpoint; `False` runs the legacy decoder only).
4. **Secret: environment variable `STATE_TOKEN_SECRET`** (hex, e.g. `openssl rand -hex 16`) - seals the conversation state tokens. When unset, a random secret is used per process and state tokens do **not** survive a restart. Set it before starting the service (systemd unit, wrapper script or shell profile).
5. **Generate data files**: run `python3 botCtl/makeDataFiles.py` once after checkout or whenever `rawData/` changes. It derives `botCtl/data/` files (`intents_raw.json`, `entity_*.json`, `feature_*.json`, `entity_bio.json`). The `options_*.json` files in `botCtl/data/` are hand-maintained and are never overwritten. `data/intents_sysone.json` and `data/category_map.json` are hand-maintained SystemOne criteria (source files, not generated).
6. **Start**: `python3 botCtl/botCtl.py` runs the Flask app on `0.0.0.0:11354` (development mode; behind gunicorn/uwsgi for production). Optional: a `botCtl/startModels.sh` is executed in the background on the first request of a fresh session, if present (log: `botCtl/startModels.log`).
7. **Reverse proxy** - external exposure is via Apache proxy settings pointing at port 11354.
8. **Verify** - `python3 botCtl/sysone_test.py` (needs `SYSONE_KEY` or a non-default `SYSONE_URL`, hits the live endpoint) and `python3 botApiTest.py --scenario all` (spawns its own service child with a generated secret).

Runtime state: `botCtl/bot_history.db` (SQLite message log, auto-created).

### Hardening (security analysis 2026-10-24)

Built into the service:

  * **Turn rate limit** - max 30 turns per session per minute (env
    `TURN_RATE_LIMIT`, 0 = off), HTTP 429 above. Each turn spends external
    LLM/SystemOne credits, so unthrottled input can drain the account quota.
    Complement with Apache-side throttling (`mod_evasive` /
    `mod_security`) and a `LimitRequestBody`.
  * **Session format guard** - `session` must match `^[A-Za-z0-9_-]{1,64}$`
    (UUIDs as issued by the server qualify); anything else is a 400.
  * **State token expiry** - tokens carry a sealed issuance timestamp and
    expire after 7 days (env `STATE_TOKEN_MAX_AGE` seconds, 0 = off).
    Tokens without a timestamp (pre-feature) remain valid.
  * **History retention** - `bot_history.db` rows older than 90 days are
    purged at startup and once a day (env `HISTORY_RETENTION_DAYS`, 0 = off).
  * **No Werkzeug debugger** - `botCtl.py -d` only enables verbose logging;
    the Flask dev server always runs with `debug=False, use_reloader=False`.
    For production prefer gunicorn:
    `gunicorn -w 2 -b 127.0.0.1:11354 botCtl:app` (bind localhost only,
    TLS at the proxy), or keep `botCtl.py` with the port firewalled off.
  * **External calls bounded** - the lupo-cloud measurement fetch uses a
    10s timeout and answers "unavailable" on failure; LQI values outside
    1..6 are treated as missing data.

Data protection (DSGVO) notes:

  * `bot_history.db` stores user message texts and the full conversation
    context in plaintext, plus session ids. It lives next to the code with
    OS-level access control only - restrict file permissions accordingly.
  * User input is sent to third-party services: the embedding/chat LLM
    (entity arbiter, follow-up gate on the legacy path) and the SystemOne
    decision endpoint (routing; GreenPT EU endpoint by default, or a
    self-hosted endpoint such as local Ollama). This must be covered by the
    privacy notice of the site and a data processing agreement per provider.
  * `botCtl/private.py` holds API keys - keep it mode 600 and out of backups
    that are not encrypted.

Forward-looking: the frontend renders HTML from bot utterances (e.g. `<br/>`).
Today every string that reaches an answer comes from curated data or server
generated options. **When the planned LLM-formulated answers land, their
output MUST be escaped or HTML-whitelisted before rendering**, or prompt
injection becomes stored XSS.


# State Token Security

The conversation context (intent, entity, feature, focus, options) is **not**
stored server-side. It is sealed into an opaque token the client carries
between requests (`botCtl/botQa.py`):

```
token = "v1." + base64url( nonce(12) || ciphertext )

key   = HMAC-SHA256(STATE_TOKEN_SECRET, session_id)   # AES-256 key per session
aad   = session_id                                    # bound as associated data
ct    = AES-256-GCM(key, nonce, json(ctx), aad)       # auth tag included
```

Four protection layers:

1. **Confidentiality** - the client cannot read the context (AES-256-GCM).
2. **Integrity** - the GCM auth tag: any modified bit in the token fails
   decryption with `InvalidTag`; tampering is infeasible without the secret.
3. **Session binding** - the key is derived per session *and* the session id
   is bound as AAD, so session A's token is doubly useless in session B.
4. **Server secret** - `STATE_TOKEN_SECRET` (see Deploy). Never set it from
   client input. Without it a random per-process secret is used and tokens
   do not survive a restart (warning printed at startup).

Rejection policy: a missing, tampered, malformed, or foreign-session token
answers **HTTP 400** - there is no plaintext fallback
(`state-token-encryption` spec in `botCtl/openspec/specs/`).

The API test suite verifies these properties:
`botApiTest.py --scenario tampered` flips one character in a valid token and
expects 400; `--scenario foreign` sends session A's token under session B
(expects 400); `--scenario restart_resume` proves that with the *same*
`STATE_TOKEN_SECRET` a session resumes after a restart, while
`--scenario restart_no_secret` proves that without a secret it does not
(400 on the stale token).


# Status

## Metis routing (2026-10-24, branch `metis-decision-routing`)

> **Renamed 2026-10-24**: the routing layer is now **SystemOne routing**
> (`botSystemone.py`, `SystemoneRouter`, `SYSONE_*` config, `sysone_test.py`,
> `data/intents_sysone.json`); OpenSpec change `rename-metis-to-systemone`.
> The GreenPT decision model `metis` remains the default; other
> SystemOne-compatible endpoints (e.g. a local Ollama server serving
> `d3:2b`) are configurable via `SYSONE_URL`/`SYSONE_MDL`. The section below
> is the original feature record under the old naming.

New experimental intent routing using the GreenPT decision model **Metis**
(`POST /v1/systemone`, EU endpoint, no text generation, probabilities only).
It replaces the embedding decoder + LLM arbiter + LLM follow-up gate with
one forward pass per turn:

  * **Stage 1** decides the coarse intent (17 options: smalltalk, beleidigung,
    bio_frage, nazka_info, anreise, ausstellungen, kinder, aktivitaeten,
    essen, hunde, fuehrungen, barrierefreiheit, toiletten, wetter, messdaten,
    expertise, fallback) *jointly* with three yes/no transition questions
    (followup, new_topic, unrelated) about the same turn.
  * **Stage 2** (bio questions only) decides the answer feature category:
    *verhalten* (behaviour: Nahrung, Fortpflanzung, Lebensraum, Feinde, Rufe
    ...), *eigenschaften* (properties: Größe, Gewicht, Erkennungsmerkmale,
    Blütezeit ...) or *beschreibung* (description: Name, Name_sci, Familie,
    Klasse, Ökologische Bedeutung ...). An undecided answer (top-2 margin
    < 0.10 or confidence < 0.35) merges all categories and offers 2-3
    feature options. Media rule: audio for verhalten, otherwise image.
  * The 30 old `tp_*`/`tiere_*`/`pflanzen_*` feature intents collapse into
    the single `bio_frage` intent; the German entity search (stem + fuzzy
    matcher) stays in place as entity resolver and arbiter.
  * Follow-ups ("und wie schwer wird er?") reuse the conversation focus
    without an entity search; a confirmed topic switch clears it.

New files: `botCtl/botMetis.py` (client + thresholds), `botCtl/data/intents_metis.json`
(criteria descriptions, `bio_frage` record), `botCtl/data/category_map.json`
(column groups), `botCtl/metis_test.py` (smoke test, live endpoint).
The derived `data/entity_bio.json` (entity names across all three domains)
comes from `makeDataFiles.py`.

### Metis vs. legacy — measured comparison

  * Intent accuracy: **89.3%** over 882 real example questions from the old
    93 intents (4 batch iterations; "fair" score counting defensible
    alternative routings ~90%). Legacy was never measured end-to-end.
  * Follow-up vs. topic switch separated cleanly (0.96 / 0.92) - the weakest
    part of the legacy pipeline (separate LLM gate).
  * Latency/cost: 1 Metis call (non-bio) / 2 calls (bio) at ~70-200ms and
    ~400 input tokens, vs. embedding call + mid-band chat-LLM arbiter on the
    legacy path.
  * Maintenance: 93 intents + 3687 vectors vs. 17 criteria texts - new cases
    are one German sentence, not a re-embedding.
  * Known blind spot: "dein Lieblingstier" (bot persona favourites) decodes
    to bio_frage (31/882); legacy vectors caught it. To be answered by the
    LLM persona path later.
  * Trade-offs: external EU service (rate limit 600 req/5min per account,
    fallback to legacy on outage is built in), model behaviour can change
    with `impact.version`, and the 89.3% were measured on legacy-authored
    example questions - real user behaviour is the missing measurement.

### Activation

(Original activation record; keys are now `SYSONE_ROUTING`/`SYSONE_KEY`,
test is now `sysone_test.py`.)

`METIS_ROUTING` in `private.py` (default: **on** when `METIS_KEY` exists;
set `METIS_ROUTING = False` for the legacy decoder). Any Metis error falls
back to the legacy path per turn. Verified: `metis_test.py` 20/20 PASS and
`botApiTest.py --scenario all` fully green (11 scenarios, 16 entity samples)
against the Metis-on server. OpenSpec change: `openspec/changes/add-metis-decision-routing/`.
Recommendation: keep the legacy path deployable, observe `bot_history.db`
(`llm_used`, intent names) during a trial period before retiring anything.

## Current (2026-02-02)

### Front-End
Companion frontent: https://github.com/CodeforKarlsruhe/auenlaend 

### Backend

This repo: https://github.com/CodeforKarlsruhe/auenBot

Required definition files are in folder **rawData**

 * intents.json: List of intents with text examples, (optional) output and optional properties (to be documented).
 * tiere_pflanzen_auen.json: The entity database 
 * intent_vectors.json: Embedding vectors


All functionality is in folder **botCtl**

  * botCtl.py: the wrapper to run the backend app. External exposure is via apache proxy settings 
  * botDecoder.py: finds target intent or fallback from user input and defined intents
  * botIntents.py: Handles specific intent via handlers and derived/detected properties. Eventual matching is done via *rapidfuzz/fuzzywuzzy* string matching (as in previous implementaion)
  * botActions.py: handles any actions needed for intent output generation, e.g. retrive measurement values from LUBW api
  * botVectors.py: loads and queries intent vectors using LLM ebedding model *bge-m3*. **This is different to the previous implementation**


The script *botCtl/makeDataFiles.py* is used to derive a more compact intent description file (without the text samples) in botCtl/data/intents_raw.json and a series of files with feature and entity lists for matching. 

**NB** The options_\<something\>.json files in the folder *botCtl/data* cannot be generated automatically but must be manually edited (as in repo).

**botCtl.py** handles the frontend communication and stores a record of received and transmitted messages in *bot_history.db* (sqlite).


### Localization
Is partially prepared for, defaults to German (de).


### Specifications 

 * Intents: list of intent items like so:
 >   {
    "id":"63b6a1f6d9d1941218c5c7c4",
    "intent":"begrüßung",
    "utter":"Hi, ich bin KarlA, deine Chatbot-Rangerin😊!\nIch freue mich darauf, mit dir zu chatten.<br \/>Als Chatbot-Rangerin beantworte ich Fragen rund um den Lebensraum \"Karlsruher Rheinauen\". Auch Fragen zum Naturschutzzentrum Karlsruhe-Rappenwört kann ich beantworten.<br \/>\nEin paar Hinweise, bevor es losgeht: In der Natur begegnen wir uns alle auf Augenhöhe, daher benutze ich das \"Du\". Ich bitte dich, mir keine persönlichen Daten wie deinen Namen oder deine Adresse zu verraten. Wenn du mehr über mich wissen möchtest: Klicke auf Chatbot-Rangerin KarlA2.\nSo - jetzt ab zu spannenden Naturthemen! Hast du eine Frage oder darf ich dir etwas zu meinen Lieblingsthemen erzählen?",
    "intent_en":"greeting",
    "intent_fr":"bienvenue",
    "intent_de":"begrüßung",
    "utter_de":"Hi, ich bin KarlA, deine Chatbot-Rangerin😊!\nIch freue mich darauf, mit dir zu chatten.<br \/>Als Chatbot-Rangerin beantworte ich Fragen rund um den Lebensraum \"Karlsruher Rheinauen\". Auch Fragen zum Naturschutzzentrum Karlsruhe-Rappenwört kann ich beantworten.<br \/>\nEin paar Hinweise, bevor es losgeht: In der Natur begegnen wir uns alle auf Augenhöhe, daher benutze ich das \"Du\".\n  Ich bitte dich, mir keine persönlichen Daten wie deinen Namen oder deine Adresse zu verraten. Wenn du mehr über mich wissen möchtest: Klicke auf Chatbot-Rangerin KarlA2.\nSo - jetzt ab zu spannenden Naturthemen! Hast du eine Frage oder darf ich dir etwas zu meinen Lieblingsthemen erzählen?",
    "utter_en":"Hi, I'm KarlA, your Chatbot Ranger! 😊 I'm looking forward to chatting with you. As a Chatbot Ranger, I answer questions about the habitat \"Karlsruher Rheinauen\" and also about the Nature Conservation Center in Karlsruhe-Rappenwört.\nA few pointers before we get started: In nature, we all meet on equal footing, so I use the \"Du\".\nI ask you not to reveal any personal data such as your name or address. If you want to know more about me, click here: Chatbot Rangerin KarlA2.\nSo - off to exciting nature topics! Do you have a question or should I tell you something about my favorite topics?",
    "utter_fr":"Bonjour, je suis KarlA, votre guide chatbot pour le site \"Karlsruher Rheinauen\" ! Je suis impatiente de discuter avec vous. En tant que guide chatbot, je répondrai à toutes vos questions sur le site du \"Karlsruher Rheinauen\" et sur le centre de conservation de Karlsruhe-Rappenwört.\nQuelques conseils avant de commencer : dans la nature, nous sommes tous sur un pied d'égalité, c'est pourquoi je vais vous parler du \"tu\". N'oubliez pas de ne pas partager vos informations personnelles comme votre nom ou votre adresse. Si vous voulez en savoir plus sur moi, rendez-vous sur Chatbot-Rangerin KarlA2.\nAlors, plongeons dans des sujets naturels passionnants ! Avez-vous une question ou souhaite-t-il que je vous parle de mes sujets préférés ?",
    "link":{
      "title":"Chatbot-Rangerin KarlA2",
      "url":"https:\/\/nazka.de\/chatbot-karla"
    },
    "action":null,
    "alias_de":null,
    "alias_en":null,
    "alias_fr":null,
    "handler":"default",
    "requires":null,
    "options":null,
    "provides":null
  },

*id* and *intent* are the main keys to identify an intent. 
*Aliases* are descriptive texts for LLM prompting in case LLM resolution of target intent is required.  
*handler* specifies the handling mechanism, currently **default** for a simple intent which just produces an output. Or **complete** for intents which need to assemble property values (entity, feature, type) from user input or provided values. 

Example: 
>   "requires":"entity,feature",
    "options":null,
    "provides":{
      "feature":"definition"
    }

The completion handler requires *entity* and *feature*. From the intent decoding we already get the feature *definition*. So we need to search only for a matching *entity* in the user input. This allows to use generic handling for a number of intents instead of individual handlers for each intent. More (complex) handlers can easily be added, if required.  




 * Database: list of items like so:
 >   {
    "Erkennungsmerkmale":"Die etwa 28mm große Blauschwarze Holzbiene ist unsere größte heimische Wildbienenart. Unter ihrer Last neigt sich mancher Blütenkopf nach unten.  Durch ihren blauschimmernden Panzer und die blauschwarzen Flügel ist sie nicht mit Hummeln zu verwechseln.",
    "Habitat":"Die wärmeliebende Wildbienenart wird durch die Klimaerwärmung in Deutschland immer häufiger.\nSie bewohnt sonnenbeschienene Biotope mit reichlich Totholz und großer Blütenvielfalt. Beliebte Lebensräume der Blauen Holzbiene sind strukturreiche Streuobstwiesen, naturnahe Gärten, Parkanlagen und lichte Waldränder.",
    "Fortpflanzung":"Die Holzbiene lebt nicht in einem Volk, sondern alleine. Sie wird schon früh im Jahr aktiv und beginnt mit ihren kräftigen Mundwerkzeugen eine fingerdicke, etwa 30 cm lange Brutröhre in totes Holz zu nagen. Ans Ende der Röhre trägt sie einen Nahrungsvorrat aus Pollen und Nektar ein, bis die Kammer randvoll ist und legt ein Ei darauf. Anschließend verschließt sie die Zelle mit einer Wand aus Holzspänen und Speichel. In einer Brutröhre werden um die 15 Zellen angelegt. Die fertig entwickelten Bienen fliegen bereits im Juni aus.",
    "Größe":"Die etwa 28mm große Blauschwarze Holzbiene ist unsere größte heimische Wildbienenart. Unter ihrer Last neigt sich mancher Blütenkopf nach unten.",
    "Name":"Blauschwarze Holzbiene",
    "Name_alt":"Große Blaue Holzbiene",
    "Links":[
      {
        "img":"https:\/\/upload.wikimedia.org\/wikipedia\/commons\/thumb\/c\/c6\/Abeille_charpentiere_1024.jpg\/330px-Abeille_charpentiere_1024.jpg"
      }
    ],
    "Typ":"Tier",
    "Name_sci":"Xylocopa violacea",
    "Name_eng":"carpenter bee",
    "Familie":"Echte Bienen (Apidae)",
    "Gattung":"Holzbienen (Xylocopa)",
    "Lebensraum":"Hartholzaue",
    "Klasse":"Insekten",
    "Lebensweise":null,
    "Nahrung":null,
    "Feinde":null,
    "Lebenserwartung":null,
    "Schutz":null,
    "Wissenswertes":null,
    "Blütezeit":null,
    "Verwendung":null,
    "Frucht":null,
    "Vorkommen":null,
    "Genießbarkeit":null,
    "Ökologische Bedeutung":null,
    "Giftigkeit":null,
    "Alter":null,
    "Gewicht":null,
    "Überwinterung":null,
    "Verhalten":null,
    "Paarung":null,
    "Gruppe":"Blauschwarze Holzbiene",
    "Unterklasse":"Hautflügler"
  },

Every items has the keys *Typ*, *Name* and *Name_sci*, all other keys are optional.
So far, *Name* is used for matching.



 * Intent Vectors
    Each intent comes with a list of (German) sample texts. The embedding model is used to generate a vector file **rawData/intent_vectors.json** which created the embedding vector for each sentence and maps it to the corresponding intent id. The vectors can be queried for similarity to find the best N matches for any input. **NB** This is a very simple approach. Brute force search against all vectors and a very large vector file (needs git lfs !). Production would use more efficient implementations (e.g. protobuf to get smaller file). However, with less than 10000 vectors the use of a vector database or appoximate nearist-neightbour search (like faiss) would be overkill (if not already in place).




## Initial (2025-12-30)
Analyzed dataset from LUBW

Extracted chatbot signatures for intents and responses 

Extracted data from plants, animals and Rheinauen" area

Intents also relate to access to current whether conditions and environmental data. 

Initial decoding and routing implemented in python. Input matching with rapidfuzz (text matchin library). 
Fallback to remote LLM if required.
Options for vector-search but initial version with BM25 not very helpfull. Future test with bge-m3 pending (vectors from intent samples already generated).


## RawData

### Primary files
  * tiere_pflanzen_auen.json: knowledge base. dataset for animals, plants and some Rheinauen types.
  * intents.json: intents with sample texts and utterance (if any)
  * intent_vectors.json: vectorized (bge-m3 embeddings) text samples and corresponding intent_id. **Needs git lfs**

### Original bot
  * tagsAndInstructions.json: additional info for original bot decoding and routing

### Intent detection

  1) Find "simple" intents, which ,ap to one specific answer, like *wo gibt es etwas zu essen?*
  2) Find "simple" intents which require function calling, like *wie ist der CO2 Gehalt?*
  3) Find intents which require specific information, usually acquired during multiple steps using a more or less complex context. 
    * Travel information with from,to,date
    * Entity (Tiere, Pflanzen, Auen) information using various key parameters (properties, class, type), like so (from original):

```python
    > tp_generell_extract_information(latest_msg):result_matching = process.extractOne(latest_msg, animal_categories)
  if result_matching[1] > 80
      result[0] = result_matching[0]
  else result_matching = process.extractOne(latest_msg, ["Tiere", "Pflanzen", "B\u00e4ume", "Blumen"])
      if result_matching[1] > 80
          if result_matching[0] == "Tiere":result[0] = "Tiere" 
          else result[0] = "Pflanzen" result_matching = process.extractOne(latest_msg, lr_categories)
              if result_matching[1] > 80
                  result[1] = result_matching[0]
                      current_lr = result[1]
                      return result
                      
  def tp_generell_generate_answer(entities):
  result = []
  if entities[0] == "Tiere":
      for animal in ANIMAL.keys():
          if not entities[1] or lr_categories.index(entities[1]) in ANIMAL[animal][4]:
              result.append(animal)
          elif entities[0] == "Pflanzen"
              for plant in PLANT.keys()
                  if not entities[1] or lr_categories.index(entities[1]) in PLANT[plant][4]:
                      result.append(plant)
                  else:
                      for animal in ANIMAL.keys()
                          if ANIMAL[animal][3] == str(animal_categories.index(entities[0])):
                              if not entities[1] or lr_categories.index(entities[1]) in ANIMAL[animal][4]:
                                  result.append(animal)
  return result

```



### Auxiliary, input or leftover files
  * pflanzenKeys.json: Parameters for plant descriptions 
  * tiereKeys.json: Parameters for animal descriptions 
  * taskList.json: decoded signatures. if *utter* is present, it should be used as response. Otherwise, intent should either start with *tp_*, *tiere_*, *pflanzen_*  which should then address the data from the corresponding types (or both), or with *wetter* or *messdaten*.  Reference to the few *Rheinauen* datasets has to be defined still.

## Media

### Directory of fantasy images
512*512, generated by flux-1-schnell.



## Next Steps
### Basic Bot
Create vector embedding for all intent texts. Setup database with vectors, full text and intent names. Test chatbot response to arbitrary requests. 

### Reference Data

Add data access to whether conditions, environmental data, Wikidata images and audio files, Source to be found, probably from NAZKA, or https://www.museumfuernaturkunde.berlin/de/forschung/tierstimmenarchiv. MP3 files were missing in input dataset.

