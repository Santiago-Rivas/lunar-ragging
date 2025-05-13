import time, json, psutil
from pathlib import Path
from itertools import product, combinations
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ragas import evaluate
from ragas.metrics import Faithfulness
from datasets import Dataset
from src.rag_refactored import ConnectionRAG, ingest_from_dirs

DATA_DOSSIER = Path("../data/dossiers")
DATA_HTML    = Path("../data/html")
EVENT_ID     = "moon_event"
TARGET_ID    = "1"   # user we’ll query for

DIVERSITY_K = 10
proc = psutil.Process()
records = []

# one faithfulness metric instance
faith_metric = Faithfulness()

for ch_size, ch_overlap in product([128], [64]):
    rag = ConnectionRAG(chunk_size=ch_size, chunk_overlap=ch_overlap)

    # --- indexing ---
    mem_before_idx = proc.memory_info().rss / 1024**2
    t0 = time.perf_counter()
    ingest_from_dirs(rag, DATA_DOSSIER, DATA_HTML, EVENT_ID)
    idx_time = time.perf_counter() - t0
    mem_after_idx = proc.memory_info().rss / 1024**2

    # throughput
    n_chunks = len(rag.collection.get(include=[])["ids"])
    throughput = n_chunks / idx_time if idx_time > 0 else 0

    # --- prepare retriever & dossier ---
    where = {"$and":[{"event_id":{"$eq":EVENT_ID}},{"user_id":{"$ne":TARGET_ID}}]}
    retriever = rag.collection.as_retriever(search_type="mmr",
                                            search_kwargs={"k":20,"filter":where})
    hit = rag.collection.get(where={"user_id":{"$eq":TARGET_ID}},
                             limit=1, include=["metadatas"])
    target_name    = hit["metadatas"][0].get("name","")    if hit["metadatas"] else ""
    target_dossier = hit["metadatas"][0].get("dossier","") if hit["metadatas"] else ""

    # --- querying & collect raw answer + docs ---
    mem_before_q = proc.memory_info().rss / 1024**2
    latencies = []
    raw_answer = None
    all_docs = None

    for i in range(20):
        # retrieve context chunks
        docs = retriever.invoke(target_dossier)
        if i == 0:
            all_docs = docs

        # build and send prompt
        parts = [{"user_id":d.metadata["user_id"],
                  "name":   d.metadata.get("name",""),
                  "chunk":  d.page_content}
                 for d in docs[:10]]
        prompt = rag.build_prompt(TARGET_ID, target_name, target_dossier, parts, k=5)

        t1 = time.perf_counter()
        # get raw text for faithfulness
        import openai
        resp = openai.OpenAI().chat.completions.create(
            model=rag.CHAT_MODEL,
            temperature=0.2,
            messages=[{"role":"user","content":prompt}],
        )
        text = resp.choices[0].message.content.strip()
        latencies.append(time.perf_counter() - t1)

        if raw_answer is None:
            raw_answer = text
    mem_after_q = proc.memory_info().rss / 1024**2

    # --- redundancy & diversity as before ---
    top = all_docs[:DIVERSITY_K]
    uids = [d.metadata["user_id"] for d in top]
    redundancy = 1 - len(set(uids)) / len(uids) if uids else 0

    texts = [d.page_content for d in top]
    if texts:
        embs = rag.embeddings.embed_documents(texts)
        sims = cosine_similarity(embs)
        dists = [1 - sims[i,j] for i,j in combinations(range(len(sims)),2)]
        diversity = sum(dists) / len(dists)
    else:
        diversity = 0

    # --- faithfulness via RAGAS.evaluate ---
    sample = {
        "question": [f"Suggest matches for user {TARGET_ID}"],
        "answer":   [raw_answer],
        "contexts": [[d.page_content for d in all_docs[:10]]],
    }
    ds = Dataset.from_dict(sample)
    df_eval = evaluate(dataset=ds, metrics=[faith_metric]).to_pandas()
    faith_score = df_eval.loc[0, "faithfulness"]

    # --- p95 latency ---
    p95 = round(1000 * sorted(latencies)[int(0.95*len(latencies))], 1)

    # --- Context Precision proxy ---
    # 1) embedding de la query
    query_emb = rag.embeddings.embed_query(target_dossier)

    # 2) embeddings de los top-K docs
    top_docs = all_docs[:DIVERSITY_K]
    doc_texts = [d.page_content for d in top_docs]
    doc_embs  = rag.embeddings.embed_documents(doc_texts)

    # 3) similitud coseno
    sims = cosine_similarity([query_emb], doc_embs)[0]  # array de longitud K

    # 4) define un umbral de “relevancia”
    THRESH = 0.75
    n_relevant = sum(1 for s in sims if s >= THRESH)
    context_precision = n_relevant / len(sims) if sims.size else 0

    records.append({
        "chunk_size":     ch_size,
        "chunk_overlap":  ch_overlap,
        "num_chunks":     n_chunks,
        "indexing_sec":   round(idx_time,2),
        "throughput_cps": round(throughput,1),
        "mem_before_idx": round(mem_before_idx,1),
        "mem_after_idx":  round(mem_after_idx,1),
        "mem_before_q":   round(mem_before_q,1),
        "mem_after_q":    round(mem_after_q,1),
        "p95_latency_ms": p95,
        "redundancy":     round(redundancy,3),
        "diversity":      round(diversity,3),
        "faithfulness":   round(faith_score,3),
        "context_precision": round(context_precision, 3),
    })

df = pd.DataFrame(records)
df.to_csv("chunk_bench_with_faithfulness.csv", index=False)
print("✓ Completed → chunk_bench_with_faithfulness.csv")
