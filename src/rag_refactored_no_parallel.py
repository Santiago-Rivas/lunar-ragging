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
from typing import List, Protocol, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import chromadb
import openai
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

# ---------- configuration --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-MiniLM-L3-v2")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
CHROMA_BASE_DIR = os.getenv("CHROMA_BASE_DIR", "./chroma_dbs")
COLLECTION_NAME = "participants"
DEFAULT_METRICS_FILE = "metrics_results.csv"

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
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 chroma_base_dir: str = CHROMA_BASE_DIR, metrics_file: Optional[str] = None, 
                 mmr_lambda: float = 0.5):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = Path(chroma_base_dir) / f"cs{chunk_size}_co{chunk_overlap}"
        self.persist_directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        self.metrics_file = metrics_file
        self.mmr_lambda = mmr_lambda

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

        self.splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",  # OpenAI's encoding 
            )
        
        # Initialize CSV file with headers if specified
        if self.metrics_file:
            metrics_path = Path(self.metrics_file)
            # Create file with headers if it doesn't exist
            if not metrics_path.exists():
                with open(metrics_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['chunk_size', 'chunk_overlap', 'event_id', 'user_id', 'k', 'mmr_lambda', 'faithfulness', 'relevancy'])
                logging.info(f"Created metrics file: {self.metrics_file}")

    def _get_collection(self) -> Chroma:
        try:
            return Chroma(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,  # ← object with both methods
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
            # Remove CONFIDENTIAL DOSSIER header and anything until newline
            short_text = re.sub(r'CONFIDENTIAL DOSSIER.*?\n', '', doc.text)
            for chunk in self.splitter.split_text(short_text):
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
                len(docs),
            )
        else:
            logging.warning("No texts to index!")

    # ---- retrieval & suggestion -------------------------------------------
    def suggest_connections(self, user_id: str | list[str], event_id: str, k: int = 5,
                            evaluate_faith: bool = False, visualize: bool = False) -> dict:
        # Handle both single user_id and list of user_ids
        user_ids = [user_id] if isinstance(user_id, str) else user_id

        results = {}

        for uid in user_ids:
            # combine the two predicates under a single $and operator
            where_filter = {
                "$and": [
                    {"event_id": {"$eq": event_id}},
                    {"user_id": {"$ne": uid}},
                ]
            }

            retriever = self.collection.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 20, 
                    "filter": where_filter,
                    "lambda_mult": self.mmr_lambda
                },
            )

            target_user = self._lookup_user(uid)
            target_name = ""
            target_dossier = ""
            if target_user["metadatas"]:
                target_name = target_user["metadatas"][0].get("name", "")
                target_dossier = target_user["metadatas"][0].get("dossier", "")
            
            # Chunk the target dossier and retrieve for each chunk
            if target_dossier:
                chunks = self.splitter.split_text(target_dossier)
                all_docs_with_scores = []
                
                for chunk in chunks:
                    # Use similarity_search_with_score instead of retriever.invoke to get scores
                    chunk_docs_with_scores = self.collection.similarity_search_with_score(
                        chunk, 
                        k=20, 
                        filter=where_filter
                        # Note: similarity_search_with_score doesn't support mmr directly,
                        # but we're tracking this param for the main retriever
                    )
                    all_docs_with_scores.extend(chunk_docs_with_scores)
                
                # Sort all documents by score (lower is better)
                all_docs_with_scores.sort(key=lambda x: x[1])
                
                # Deduplicate by user_id, keeping highest scoring (lowest value) for each
                user_id_to_best_doc = {}
                for doc, score in all_docs_with_scores:
                    user_id = doc.metadata.get("user_id")
                    if user_id not in user_id_to_best_doc or score < user_id_to_best_doc[user_id][1]:
                        user_id_to_best_doc[user_id] = (doc, score)
                
                # Get the top 20 documents, sorted by score
                best_docs = sorted(user_id_to_best_doc.values(), key=lambda x: x[1])[:20]
                docs = [doc for doc, _ in best_docs]
                
                # Visualize the embeddings if requested
                if visualize:
                    target_embedding = self.embeddings.embed_query(target_dossier)
                    self.visualize_embeddings(all_docs_with_scores, best_docs, uid, target_embedding)
            else:
                # Fallback to using the whole dossier if it's empty
                docs = retriever.invoke(target_dossier)

            context_items = []
            for d in docs[:10]:
                context_items.append(
                    {
                        "user_id": d.metadata["user_id"],
                        "name": d.metadata.get("name", ""),
                        "chunk": d.page_content.strip(),
                    }
                )

            prompt = self._build_prompt(uid, target_name, target_dossier, context_items, k)

            suggestions = self._ask_llm(prompt)

            result = {"suggestions": suggestions}

            # If evaluation is requested, add faithfulness score
            if evaluate_faith:
                evaluation_scores = self._evaluate_metrics(uid, target_name, target_dossier, context_items, suggestions, k, event_id)
                result["faithfulness_score"] = evaluation_scores["faithfulness"]
                result["relevancy_score"] = evaluation_scores["relevancy"]

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
    def _evaluate_metrics(self, user_id: str, user_name: str, user_dossier: str, context_items: list[dict],
                         suggestions: list[dict], k: int, event_id: str = "") -> dict:
        """Evaluate the faithfulness and relevancy of suggestions using RAGAS."""
        prompt = self._build_prompt(user_id, user_name, user_dossier, [], k)

        questions = []
        answers = []
        contexts = []
        for suggestion in suggestions:
            questions.append(prompt)
            answers.append(suggestion['reason'])
            suggestion['user_id'] = user_id
            context = [user_dossier]
            for item in context_items:
                if item['user_id'] == suggestion['user_id']:
                    context.append(item['chunk'])
            contexts.append(context)

        # Create dataset for RAGAS evaluation
        eval_data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }
        dataset = Dataset.from_dict(eval_data)

        # Evaluate metrics
        score = evaluate(dataset, metrics=[faithfulness, answer_relevancy], raise_exceptions=True)

        faithfulness_score = np.mean(score["faithfulness"])
        relevancy_score = np.mean(score["answer_relevancy"])
        
        # print(
        # f"\n\n\n EVALUATION METRICS: {score}\n\n",
        # f"\n\n Faithfulness: {faithfulness_score}\n\n",
        # f"\n\n Relevancy: {relevancy_score}\n\n",
        # f"\n\nEvaluating metrics for {user_name},{user_dossier}\n\n",
        # f"\n\n Questions: {questions}\n\n",
        # f"\n\n Answers: {answers}\n\n",
        # f"\n\n Contexts: {contexts}\n\n\n")
        
        # Write metrics to CSV if a file was specified
        if self.metrics_file:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.chunk_size,
                    self.chunk_overlap,
                    event_id,
                    user_id,
                    k,
                    self.mmr_lambda,
                    faithfulness_score,
                    relevancy_score
                ])
                logging.info(f"Saved metrics for user {user_id} to {self.metrics_file}")
        
        return {
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score
        }

    def _evaluate_faithfulness(self, user_id: str, user_name: str, user_dossier: str, context_items: list[dict],
                             suggestions: list[dict], k: int) -> float:
        """Legacy method for backward compatibility"""
        scores = self._evaluate_metrics(user_id, user_name, user_dossier, context_items, suggestions, k)
        return scores["faithfulness"]
        
    def visualize_embeddings(self, all_docs_with_scores, best_docs, user_id, target_embedding=None):
        """
        Visualize document embeddings before and after reranking using t-SNE.
        
        Args:
            all_docs_with_scores: List of (doc, score) tuples before deduplication
            best_docs: List of (doc, score) tuples after deduplication and reranking
            user_id: User ID being analyzed
            target_embedding: Optional embedding of the target document
        """
        # Create output directory
        output_dir = Path("visualization_output")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Extract embeddings and prepare data
        all_embeddings = []
        all_labels = []
        all_scores = []
        best_indices = []
        
        for i, (doc, score) in enumerate(all_docs_with_scores):
            # Get embedding from document
            embedding = self.embeddings.embed_query(doc.page_content)
            all_embeddings.append(embedding)
            all_labels.append(doc.metadata.get("user_id", "unknown"))
            all_scores.append(score)
            
            # Check if this document is in best_docs
            for best_doc, _ in best_docs:
                if doc.page_content == best_doc.page_content:
                    best_indices.append(i)
                    break
        
        # Add target embedding if provided
        if target_embedding is not None:
            all_embeddings.append(target_embedding)
            all_labels.append(f"{user_id} (target)")
            all_scores.append(0)  # Placeholder score for target
            target_idx = len(all_embeddings) - 1
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        # 2. Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # 3. Create t-SNE plot showing before and after reranking
        plt.figure(figsize=(12, 10))
        
        # Plot all documents
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.5, label='All retrieved chunks')
        
        # Highlight best documents
        plt.scatter(embeddings_2d[best_indices, 0], embeddings_2d[best_indices, 1], 
                    c='blue', alpha=0.8, s=100, label='Reranked/deduplicated chunks')
        
        # Highlight target document if provided
        if target_embedding is not None:
            plt.scatter(embeddings_2d[target_idx, 0], embeddings_2d[target_idx, 1], 
                       c='red', s=200, marker='*', label='Target user')
        
        # Add labels for all best documents (not just top 5)
        for i in best_indices:
            plt.annotate(all_labels[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'Embeddings visualization for user {user_id}')
        plt.legend()
        plt.tight_layout()
        
        # Save the t-SNE plot
        plt.savefig(output_dir / f"embeddings_tsne_{user_id}.png")
        plt.close()
        
        # 4. Create score distribution plot
        plt.figure(figsize=(10, 6))
        
        # All scores
        all_scores_array = np.array(all_scores)
        # Remove the target score which is a placeholder
        if target_embedding is not None:
            all_scores_array = all_scores_array[:-1]
        
        # Get scores for best docs
        best_scores = [all_scores_array[i] for i in best_indices]
        
        # Plot histograms
        plt.hist(all_scores_array, bins=20, alpha=0.5, label='All chunks', color='lightgray')
        plt.hist(best_scores, bins=20, alpha=0.7, label='Selected chunks', color='blue')
        
        plt.axvline(np.mean(all_scores_array), color='gray', linestyle='dashed', linewidth=1, label=f'Mean all: {np.mean(all_scores_array):.3f}')
        plt.axvline(np.mean(best_scores), color='blue', linestyle='dashed', linewidth=1, label=f'Mean selected: {np.mean(best_scores):.3f}')
        
        plt.title(f'Score Distribution Before/After Reranking for User {user_id}')
        plt.xlabel('Similarity Score (lower is better)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        
        # Save the score distribution plot
        plt.savefig(output_dir / f"score_distribution_{user_id}.png")
        plt.close()
        
        # 5. Create a plot showing user representation before/after reranking
        plt.figure(figsize=(12, 6))
        
        # Count occurrences of each user_id before reranking
        user_counts_before = {}
        for label in all_labels:
            if label == f"{user_id} (target)":  # Skip target user
                continue
            user_counts_before[label] = user_counts_before.get(label, 0) + 1
        
        # Count occurrences of each user_id after reranking
        user_counts_after = {}
        for i in best_indices:
            label = all_labels[i]
            user_counts_after[label] = user_counts_after.get(label, 0) + 1
        
        # Get top 10 users by count before reranking
        top_users = sorted(user_counts_before.items(), key=lambda x: x[1], reverse=True)[:10]
        top_user_ids = [u for u, _ in top_users]
        
        # Prepare data for bar chart
        user_counts_before_top = [user_counts_before.get(uid, 0) for uid in top_user_ids]
        user_counts_after_top = [user_counts_after.get(uid, 0) for uid in top_user_ids]
        
        # Create a grouped bar chart
        x = np.arange(len(top_user_ids))
        width = 0.35
        
        plt.bar(x - width/2, user_counts_before_top, width, label='Before reranking', color='lightgray')
        plt.bar(x + width/2, user_counts_after_top, width, label='After reranking', color='blue')
        
        plt.xlabel('User ID')
        plt.ylabel('Number of chunks')
        plt.title(f'Number of chunks per user before/after reranking for User {user_id}')
        plt.xticks(x, top_user_ids, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the user representation plot
        plt.savefig(output_dir / f"user_representation_{user_id}.png")
        plt.close()
        
        logging.info(f"Visualizations saved to visualization_output/ directory for user {user_id}")
        logging.info(f"- t-SNE plot: embeddings_tsne_{user_id}.png")
        logging.info(f"- Score distribution: score_distribution_{user_id}.png") 
        logging.info(f"- User representation: user_representation_{user_id}.png")


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
                name=name_map.get(user_id, ""),  # ← new field
            )
        )
        html_path = html_dir / f"{user_id}.html"
        if html_path.exists():
            html_doc = HTMLResearchDoc(
                    user_id=user_id,
                    event_id=event_id,
                    html_path=html_path,
                    name=name_map.get(user_id, ""),
                )
            
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
    sug.add_argument("--mmr_lambda", type=float, default=0.5, help="Lambda multiplier for MMR (0-1, higher values prioritize relevance over diversity)")
    sug.add_argument("--visualize", action="store_true", help="Visualize embeddings")
    sug.add_argument("--metrics_file", type=str, default=DEFAULT_METRICS_FILE, help="CSV file to save metrics")

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
        metrics_file = args.metrics_file if args.evaluate else None
        rag = ConnectionRAG(
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap,
            metrics_file=metrics_file,
            mmr_lambda=args.mmr_lambda
        )
        results = rag.suggest_connections(args.user_id, args.event_id, args.k, args.evaluate, args.visualize)

        # for user_id, user_results in results.items():
        #     print(f"\n--- Results for user_id: {user_id} ---")
        #     if args.evaluate:
        #         print(f"Faithfulness Score: {user_results['faithfulness_score']}")
        #         print(f"Relevancy Score: {user_results['relevancy_score']}")
        #         print("Suggestions:")

        #     print(json.dumps(user_results['suggestions'], indent=2, ensure_ascii=False))
            
        if args.evaluate:
            logging.info(f"Metrics saved to {args.metrics_file}")

if __name__ == "__main__":
    main()
