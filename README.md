# Connection RAG Pipeline

Refactored Retrieval‑Augmented Generation (RAG) engine that cross‑links **dossiers**, **HTML research** and **conversation summaries** to suggest meaningful connections between event participants.

---

## ✨ Key Features

* **Multi‑source ingestion** – Markdown dossiers, raw HTML pages and convo summaries become a unified knowledge base.
* **Chunk‑level embeddings** – Configurable size/overlap (default 1 kB / 128 chars) with OpenAI `text‑embedding‑3‑small`.
* **Persistent vector store** – Uses ChromaDB on‑disk; reruns merely append.
* **MMR retrieval + LLM re‑ranking** – Maximal‑marginal‑relevance surfaces diverse yet relevant chunks.
* **Name‑aware metadata** – Each chunk keeps `user_id`, `name`, `source`, `event_id`, `dossier`.
* **CLI workflow** – `index` to ingest, `suggest` to produce JSON matches.

---

## Folder Layout

```
.
├── data/
│   ├── dossiers/
│   │   ├── 1.md          # dossier text per user (numeric id)
│   │   └── id_to_name.csv  # mapping id;name
│   └── html/
│       └── 1.html        # optional extra research per user
├── src/
│   └── rag_refactored.py  # main pipeline
└── chroma_db/             # auto‑created persistent DB (default)
```

### `id_to_name.csv`

Simple `;`‑separated file:

```
id;name
1;Lionel Messi
2;Ada Lovelace
…
```

---

## Quick Start

```bash
# 1 Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2 Set your OpenAI key (and optionally tweak models)
export OPENAI_API_KEY="sk‑…"
# export OPENAI_EMBED_MODEL="text-embedding-3-small"
# export OPENAI_CHAT_MODEL="gpt-4o-mini"

# 3 Prepare data folders (see above) and then ingest
python src/rag_refactored.py index \
  --dossier_dir data/dossiers \
  --html_dir    data/html      \
  --event_id    moon_event

# 4 Get suggestions for a participant (numeric id)
python src/rag_refactored.py suggest \
  --user_id 1 \
  --event_id moon_event
```

Output (pretty‑printed):

```json
[
  { "user_id": "4", "reason": "Both work on lunar rovers powered by edge AI." },
  …
]
```

---

## Environment Variables

| Variable             | Default                  | Description                    |
| -------------------- | ------------------------ | ------------------------------ |
| `OPENAI_API_KEY`     |  –                       | **Required** – your OpenAI key |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model                |
| `OPENAI_CHAT_MODEL`  | `gpt-4o-mini`            | Chat model for reasoning       |
| `CHROMA_DIR`         | `./chroma_db`            | Persisted vector DB path       |

Load values automatically by creating a **`.env`** file (parsed via `python‑dotenv`).

---

## Resetting the Vector DB

```bash
python src/rag_refactored.py index … --reset_db
```

Deletes `CHROMA_DIR` before ingesting.

---

## Implementation Notes

* **Embeddings** – full `OpenAIEmbeddings` object passed to `Chroma` (LangChain community wrappers) to satisfy both `embed_documents` & `embed_query`.
* **Metadata filter** – Chroma v0.4+ demands a single top‑level operator ⇒ we wrap predicates under `$and`.
* **Prompt layout** – Supplies target user’s *dossier* + *name* and JSON list of `{user_id, name, chunk}` for candidates.
* **Retries** – If the LLM drifts from JSON, the code reiterates with a stricter guard‑prompt.

---

## Converting a master JSON to dossiers (utility)

```bash
./json_to_dossiers.sh  # see scripts/ for details
```

Generates numbered `*.md` files and `id_to_name.csv`.

---

© 2025
