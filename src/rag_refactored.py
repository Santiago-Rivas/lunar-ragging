"""Refactored RAG pipeline with multi‑source ingestion (dossier + HTML research + conv summary)
-----------------------------------------------------------------
Requirements already satisfied in requirements.txt: langchain, chromadb, bs4, openai, python‑dotenv.

How to use (quick):
$ python rag_refactored.py index --dossier_dir data/dossiers --html_dir data/html  # ingest
$ python rag_refactored.py suggest --user_id lionel_messi --event_id moon_event   # get matches
$ python rag_refactored.py suggest --user_id lionel_messi --event_id moon_event --evaluate  # get matches with faithfulness evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
import uuid
import csv
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence

import chromadb
import openai
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate

# ---------- configuration --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-MiniLM-L3-v2")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
CHROMA_BASE_DIR = os.getenv("CHROMA_BASE_DIR", "./chroma_dbs")
COLLECTION_NAME = "participants"

load_dotenv()

# ---------- dataclasses -----------------------------------------------------
class BaseDoc(Protocol):
    user_id: str
    event_id: str
    text: str
    source: str  # "dossier" | "html" | "conv"


@dataclass
class DossierDoc:
    user_id: str
    event_id: str
    text: str
    name: str = ""
    source: str = "dossier"


@dataclass
class HTMLResearchDoc:
    user_id: str
    event_id: str
    html_path: Path
    name: str = ""
    source: str = "html"

    @property
    def text(self) -> str:  # lazily convert HTML to plaintext
        html = self.html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        # remove scripts/style
        for el in soup(["script", "style"]):
            el.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text


@dataclass
class ConvSummaryDoc:
    user_id: str
    event_id: str
    text: str
    source: str = "conv"


# ---------- RAG core --------------------------------------------------------
class ConnectionRAG:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, chroma_base_dir: str = CHROMA_BASE_DIR):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = Path(chroma_base_dir) / f"cs{chunk_size}_co{chunk_overlap}"
        self.persist_directory.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Using HuggingFace embeddings instead of OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'}
        )

        # modelo de chat (antes era global)
        self.CHAT_MODEL = CHAT_MODEL
        # 2️⃣  hand that object to Chroma
        self.collection = self._get_collection()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # Use instance attribute
            chunk_overlap=self.chunk_overlap,  # Use instance attribute
            separators=["\n\n", "\n", " "],
        )

    def _get_collection(self) -> Chroma:
        try:
            return Chroma(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,   # ← object with both methods
            )
        except ValueError:
            return Chroma.from_documents(
                [],
                self.embeddings,
                client=self.client,
                collection_name=COLLECTION_NAME,
            )

    # ---- ingestion ---------------------------------------------------------
    def add_documents(self, docs: Sequence[BaseDoc]):
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []

        for doc in docs:
            for chunk in self.splitter.split_text(doc.text):
                ids.append(str(uuid.uuid4()))
                texts.append(chunk)
                metas.append(
                    {
                        "user_id": doc.user_id,
                        "event_id": doc.event_id,
                        "source": doc.source,
                        "name": getattr(doc, "name", ""),
                        "dossier": getattr(doc, "text", "")
                    }
                )

        if texts:
            self.collection.add_texts(texts, ids=ids, metadatas=metas)
            logging.info(
                "Added %d chunks (docs=%d) to collection",
                len(texts),
                len({m['user_id'] for m in metas}),
            )
        else:
            logging.warning("No texts to index!")

    # ---- retrieval & suggestion -------------------------------------------
    def suggest_connections(self, user_id: str | list[str], event_id: str, k: int = 5, evaluate_faith: bool = False) -> dict:
        # Handle both single user_id and list of user_ids
        user_ids = [user_id] if isinstance(user_id, str) else user_id
        
        results = {}
        
        for uid in user_ids:
            # combine the two predicates under a single $and operator
            where_filter = {
                "$and": [
                    {"event_id": {"$eq": event_id}},
                    {"user_id":  {"$ne": uid}},
                ]
            }

            retriever = self.collection.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 20, "filter": where_filter},
            )

            target_user = self._lookup_user(uid)
            target_name = ""
            target_dossier = ""
            if target_user["metadatas"]:
                target_name = target_user["metadatas"][0].get("name", "")
                target_dossier = target_user["metadatas"][0].get("dossier", "")
            docs = retriever.invoke(target_dossier)

            context_items = []
            for d in docs[:10]:
                context_items.append(
                    {
                        "user_id": d.metadata["user_id"],
                        "name":    d.metadata.get("name", ""),
                        "chunk":   d.page_content.strip(),
                    }
                )

            prompt = self._build_prompt(uid, target_name, target_dossier, context_items, k)
            
            suggestions = self._ask_llm(prompt)
            
            result = {"suggestions": suggestions}
            
            # If evaluation is requested, add faithfulness score
            if evaluate_faith:
                faith_score = self._evaluate_faithfulness(uid, target_name, target_dossier, context_items, suggestions, k)
                result["faithfulness_score"] = faith_score

            results[uid] = result
            
        return results
        
    def build_prompt(
        self,
        user_id: str,
        user_name: str,
        user_dossier: str,
        context_items: list[dict],
        k: int,
    ) -> str:
        return self._build_prompt(user_id, user_name, user_dossier, context_items, k)
        
    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _build_prompt(
        user_id: str,
        user_name: str,
        user_dossier: str,
        context_items: list[dict],
        k: int,
    ) -> str:
        prompt = f"""You are an expert networking assistant.
Given the following context chunks about other participants in the same event, return a JSON list of the top {k} people that should match with user '{user_id}' ({user_name}).

This is the dossier of {user_id}:
{user_dossier}

Each item must be an object {{"user_id": str, "reason": str}}.
Only output valid JSON without markdown fences."""

        if context_items:
            prompt += f"\nContext (JSON):\n{json.dumps(context_items, ensure_ascii=False, indent=2)}"

        return textwrap.dedent(prompt)

    def ask_llm(self, prompt: str) -> list[dict]:
        return self._ask_llm(prompt)

    def _ask_llm(self, prompt: str) -> list[dict]:
        client = openai.OpenAI()
        for attempt in range(2):
            completion = client.chat.completions.create(
                model=self.CHAT_MODEL,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            content = completion.choices[0].message.content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logging.warning("Attempt %d: model returned non‑JSON, retrying", attempt + 1)
                prompt = (
                    "You did not follow the JSON format. Please output *only* the JSON list as specified, no markdown fences.\n\n"
                    + prompt
                )
        # fallback
        return [{"raw_output": content}]

    def _lookup_user(self, user_id: str) -> str:
        """Return the name stored in Chroma metadata for this user_id (or '')."""
        try:
            return self.collection.get(
                where={"user_id": {"$eq": user_id}},
                limit=1,
            )
        except Exception:
            pass
        return {}

    # ---- evaluation --------------------------------------------------------
    def _evaluate_faithfulness(self, user_id: str, user_name: str, user_dossier: str, context_items: list[dict], suggestions: list[dict], k: int) -> float:
        """Evaluate the faithfulness of suggestions using RAGAS."""
        # Format contexts and answers for RAGAS evaluation
        contexts = []
        for item in context_items:
            contexts.append(item["chunk"])

        answers = ""
        for suggestion in suggestions:
            answers += f"{suggestion['reason']}\n"

        prompt = self._build_prompt(user_id, user_name, user_dossier, [], k)
        questions = [prompt]
        
        # Create dataset for RAGAS evaluation
        eval_data = {
            'question': questions,
            'answer': [answers],
            'contexts': [contexts],
        }
        
        dataset = Dataset.from_dict(eval_data)
        
        # Evaluate faithfulness
        score = evaluate(dataset, metrics=[faithfulness], raise_exceptions=False)
        
        return score.to_pandas()["faithfulness"].iloc[0]


# ---------- CLI ------------------------------------------------------------

def ingest_from_dirs(rag: ConnectionRAG, dossier_dir: Path, html_dir: Path, event_id: str):
    docs: list[BaseDoc] = []
    name_map = _load_name_map(dossier_dir)

    for dossier_file in dossier_dir.glob("*.md"):
        user_id = dossier_file.stem
        text = dossier_file.read_text(encoding="utf-8")
        docs.append(
            DossierDoc(
                user_id=user_id,
                event_id=event_id,
                text=text,
                name=name_map.get(user_id, ""),     # ← new field
            )
        )
        html_path = html_dir / f"{user_id}.html"
        if html_path.exists():
            docs.append(
                HTMLResearchDoc(
                    user_id=user_id,
                    event_id=event_id,
                    html_path=html_path,
                    name=name_map.get(user_id, ""),
                )
            )
    rag.add_documents(docs)


def _load_name_map(dossier_dir: Path) -> dict[str, str]:
    """
    Expects a file  <dossier_dir>/id_to_name.csv  with header  id;name
    Returns { "1": "Lionel Messi", "2": "Ada Lovelace", … }.
    """
    csv_path = dossier_dir / "users.csv"
    name_map: dict[str, str] = {}
    if not csv_path.exists():
        logging.warning("Missing id_to_name.csv - names will be empty")
        return name_map

    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            name_map[row["id"]] = row["name"]
    return name_map

def main():
    parser = argparse.ArgumentParser(description="Connection RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index")
    idx.add_argument("--dossier_dir", type=Path, required=True)
    idx.add_argument("--html_dir", type=Path, required=True)
    idx.add_argument("--event_id", type=str, default="moon_event")
    idx.add_argument("--reset_db", action="store_true")
    idx.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    idx.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    sug = sub.add_parser("suggest")
    sug.add_argument("--user_id", nargs="+", required=True, help="One or more user IDs to get suggestions for")
    sug.add_argument("--event_id", default="moon_event")
    sug.add_argument("--k", type=int, default=5)
    sug.add_argument("--evaluate", action="store_true", help="Evaluate faithfulness of suggestions")
    sug.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    sug.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    args = parser.parse_args()

    if args.cmd == "index":
        db_path = Path(CHROMA_BASE_DIR) / f"cs{args.chunk_size}_co{args.chunk_overlap}"
        if args.reset_db and db_path.exists():
            import shutil
            shutil.rmtree(db_path)
            logging.info("Reset Chroma DB at %s", db_path)
        rag = ConnectionRAG(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        ingest_from_dirs(rag, args.dossier_dir, args.html_dir, args.event_id)
        logging.info("Indexing complete.")

    elif args.cmd == "suggest":
        rag = ConnectionRAG(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        results = rag.suggest_connections(args.user_id, args.event_id, args.k, args.evaluate)
        
        for user_id, user_results in results.items():
            print(f"\n--- Results for user_id: {user_id} ---")
            if args.evaluate:
                print(f"Faithfulness Score: {user_results['faithfulness_score']}")
                print("Suggestions:")
            
            print(json.dumps(user_results['suggestions'], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
