"""
rag_connections.py

Python module implementing a simple Retrieval‑Augmented Generation (RAG) system
that suggests networking connections at an event.

Dependencies
------------
    pip install chromadb sentence-transformers openai python-dotenv

Usage
-----
    python rag.py --json path/to/your/members.json
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

import openai


@dataclass
class Dossier:
    """A lightweight container for a user dossier."""

    user_id: str
    event_id: str
    text: str  # free‑form description of the attendee and their interests


class ConnectionRAG:
    """RAG pipeline for recommending people to meet at an event."""

    def __init__(
        self,
        collection_name: str = "dossiers",
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # 1️⃣  Initialise ChromaDB (persistent on disk ‑ change `path` as needed)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # 2️⃣  Embedding function (swap in sentence‑transformers if you prefer)
        if os.environ.get("OPENAI_API_KEY"):
            os.environ["CHROMA_OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
        
        self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=embedding_model,
        )
        
        # Create or get collection with the embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embed_fn
        )

    # ---------------------------------------------------------------------
    # INGESTION
    # ---------------------------------------------------------------------
    def add_dossiers(self, dossiers: List[Dossier]) -> None:
        """Add (or upsert) dossiers into the vector store."""
        ids = [f"{d.event_id}:{d.user_id}" for d in dossiers]
        texts = [d.text for d in dossiers]
        metadatas = [
            {"event_id": d.event_id, "user_id": d.user_id} for d in dossiers
        ]
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )

    # ---------------------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------------------
    def retrieve_similar(
        self,
        user_id: str,
        event_id: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return *other* attendees' dossiers most similar to the query user."""
        # Fetch the dossier for the querying user (this acts as the query text).
        query_results = self.collection.get(
            where={"$and": [
                {"event_id": {"$eq": event_id}},
                {"user_id": {"$eq": user_id}}
            ]},
            limit=1,
            include=["documents"],
        )
        if not query_results["documents"]:
            raise ValueError(
                f"Dossier not found for user '{user_id}' in event '{event_id}'"
            )

        query_text = query_results["documents"][0]

        # Similarity search (filter by event, exclude the querying user later).
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k + 1,  # +1 so we can drop the query user itself
            where={"event_id": {"$eq": event_id}},
            include=["documents", "metadatas", "distances"],
        )

        # Flatten results and filter out the querying user
        candidates: List[Dict[str, Any]] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta["user_id"] != user_id:
                candidates.append({"text": doc, "meta": meta})
            if len(candidates) == top_k:
                break
        return candidates

    # ---------------------------------------------------------------------
    # PROMPT CONSTRUCTION
    # ---------------------------------------------------------------------
    def _build_prompt(
        self,
        user_dossier: str,
        candidate_dossiers: List[Dict[str, Any]],
        top_n: int,
    ) -> str:
        prompt = (
            "You are an expert networking assistant at an industry event. "
            "You will be provided with the detailed dossier of one attendee (the USER) "
            "and ten dossiers of other attendees (the CANDIDATES).\n\n"  # noqa: E501
            "Based on the dossiers, recommend exactly five candidates the USER should meet. "
            "Rank them 1‑5 and give a concise justification (one sentence).\n\n"
        )
        prompt += "USER DOSSIER:\n" + user_dossier + "\n\n"
        prompt += "CANDIDATE DOSSIERS:\n"
        for i, c in enumerate(candidate_dossiers, 1):
            prompt += (
                f"[{i}] user_id: {c['meta']['user_id']}\n" + c["text"] + "\n\n"
            )
        prompt += (
            f"Return the result as valid JSON ‑ a list of exactly {top_n} objects "
            "with fields 'user_id' and 'reason'."
        )
        return prompt

    # ---------------------------------------------------------------------
    # LLM CALL
    # ---------------------------------------------------------------------
    def suggest_connections(
        self,
        user_id: str,
        event_id: str,
        top_k_candidates: int = 10,
        top_n_recommendations: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> List[Dict[str, str]]:
        """Retrieve candidates and obtain LLM‑ranked connection suggestions."""
        # 1. Retrieve candidates from Chroma
        candidates = self.retrieve_similar(user_id, event_id, top_k_candidates)

        # 2. Fetch the querying user's dossier text again (for the prompt)
        user_doc = self.collection.get(
            where={"$and": [
                {"event_id": {"$eq": event_id}},
                {"user_id": {"$eq": user_id}}
            ]},
            limit=1,
            include=["documents"],
        )["documents"][0]

        prompt = self._build_prompt(user_doc, candidates, top_n_recommendations)

        # 3. Call the LLM
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = response.choices[0].message.content

        # 4. Parse JSON (fallback to raw content if parsing fails)
        import json

        try:
            data = json.loads(content)
            if not isinstance(data, list) or len(data) != top_n_recommendations:
                raise ValueError
            return data  # type: ignore[return-value]
        except Exception:
            return [{"raw_output": content}]


# -------------------------------------------------------------------------
# DEMO USAGE (run `python rag_connections.py`)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG system for suggesting networking connections")
    parser.add_argument("--json", type=str, default="moon_members.json", 
                        help="Path to the JSON file containing member data (default: moon_members.json)")
    args = parser.parse_args()
    
    # Check if OpenAI API key is set in environment, otherwise prompt for it
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        if not api_key:
            print("Error: An OpenAI API key is required to run this program.")
            exit(1)
    
    # Reset the database to avoid embedding function mismatch
    import shutil
    if os.path.exists("./chroma_db"):
        print("Removing existing database to avoid embedding function mismatch...")
        shutil.rmtree("./chroma_db")
    
    # Alternative approach: Instead of deleting the database, you could use the same 
    # embedding function that was originally used to create the collection.
    # For example, to use sentence-transformers:
    # rag = ConnectionRAG(embedding_model="all-MiniLM-L6-v2")
    
    rag = ConnectionRAG(openai_api_key=api_key)

    # Load dossiers from JSON file
    import json
    import re
    
    # Load the JSON data
    json_path = args.json
    print(f"Loading members from: {json_path}")
    try:
        with open(json_path, "r") as f:
            members = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the JSON file at {json_path}")
        print(f"Current working directory: {os.getcwd()}")
        exit(1)
    
    # Create dossiers for each member
    moon_dossiers = []
    for member in members:
        # Create a user_id from the name (remove special chars, lowercase, replace spaces with underscores)
        user_id = re.sub(r'[^\w\s]', '', member['name']).lower().replace(' ', '_')
        
        # Create the dossier
        dossier = Dossier(
            user_id=user_id,
            event_id="e1",
            text=member['dossier']
        )
        moon_dossiers.append(dossier)
    
    print(f"Loaded {len(moon_dossiers)} members")
    
    # Add dossiers to the RAG system
    rag.add_dossiers(moon_dossiers)

    # Example query - suggest connections for the first member
    first_user_id = moon_dossiers[0].user_id if moon_dossiers else "u1"
    suggestions = rag.suggest_connections(first_user_id, "e1")
    print("\nSuggested connections for", first_user_id)
    print("\nSuggested connections:\n", suggestions)


