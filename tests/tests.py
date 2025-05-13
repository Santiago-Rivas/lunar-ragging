import time, psutil
from pathlib import Path
from itertools import product, combinations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
import openai

from ragas import evaluate
from ragas.metrics import Faithfulness
from src.rag_refactored import ConnectionRAG, ingest_from_dirs

DATA_DOSSIER = Path("../data/dossiers")
DATA_HTML    = Path("../data/html")
EVENT_ID     = "moon_event"
TARGET_ID    = "1"

DIVERSITY_K   = 10
proc          = psutil.Process()
records       = []
faith_metric  = Faithfulness()

# grid of parameters to sweep
chunk_sizes    = [128, 256]
chunk_overlaps = [64, 128]
k_retriever_ls = [5, 10, 20]
mmr_lambdas    = [0.2, 0.5, 0.8]
temperatures   = [0.0, 0.2, 0.7]

for ch_size, ch_overlap, k_retriever, mmr_lambda, temperature in product(
    chunk_sizes, chunk_overlaps, k_retriever_ls, mmr_lambdas, temperatures
):
    # init RAG with custom splitter
    rag = ConnectionRAG(chunk_size=ch_size, chunk_overlap=ch_overlap)

    # indexing
    mem_before_idx = proc.memory_info().rss / 1024**2
    t0 = time.perf_counter()
    ingest_from_dirs(rag, DATA_DOSSIER, DATA_HTML, EVENT_ID)
    idx_time = time.perf_counter() - t0
    mem_after_idx = proc.memory_info().rss / 1024**2

    # throughput
    n_chunks = len(rag.collection.get(include=[])["ids"])
    throughput = n_chunks / idx_time if idx_time>0 else 0

    # retriever with MMR params
    where = {"$and":[{"event_id":{"$eq":EVENT_ID}},{"user_id":{"$ne":TARGET_ID}}]}
    retriever = rag.collection.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           k_retriever,
            "filter":      where,
            "lambda_mult": mmr_lambda
        },
    )

    # fetch dossier
    hit = rag.collection.get(where={"user_id":{"$eq":TARGET_ID}},
                             limit=1, include=["metadatas"])
    target_name    = hit["metadatas"][0].get("name","")    if hit["metadatas"] else ""
    target_dossier = hit["metadatas"][0].get("dossier","") if hit["metadatas"] else ""

    # querying & collect
    mem_before_q = proc.memory_info().rss / 1024**2
    latencies, raw_answer, all_docs = [], None, None
    for i in range(20):
        docs = retriever.invoke(target_dossier)
        if i==0: all_docs = docs

        parts = [{"user_id":d.metadata["user_id"],
                  "name":   d.metadata.get("name",""),
                  "chunk":  d.page_content}
                 for d in docs[:10]]
        prompt = rag.build_prompt(TARGET_ID, target_name, target_dossier, parts, k=5)

        t1 = time.perf_counter()
        resp = openai.OpenAI().chat.completions.create(
            model=rag.CHAT_MODEL,
            temperature=temperature,
            messages=[{"role":"user","content":prompt}],
        )
        text = resp.choices[0].message.content.strip()
        latencies.append(time.perf_counter() - t1)
        if raw_answer is None:
            raw_answer = text
    mem_after_q = proc.memory_info().rss / 1024**2

    # redundancy & diversity
    top = all_docs[:DIVERSITY_K]
    uids = [d.metadata["user_id"] for d in top]
    redundancy = 1 - len(set(uids))/len(uids) if uids else 0

    texts = [d.page_content for d in top]
    if texts:
        embs = rag.embeddings.embed_documents(texts)
        sims = cosine_similarity(embs)
        dists = [1 - sims[i,j] for i,j in combinations(range(len(sims)),2)]
        diversity = sum(dists)/len(dists)
    else:
        diversity = 0

    # faithfulness
    sample = {
        "question": [f"Suggest matches for user {TARGET_ID}"],
        "answer":   [raw_answer],
        "contexts": [[d.page_content for d in all_docs[:10]]],
    }
    ds = Dataset.from_dict(sample)
    df_eval = evaluate(dataset=ds, metrics=[faith_metric]).to_pandas()
    faith_score = df_eval.loc[0, "faithfulness"]

    # context precision proxy
    query_emb = rag.embeddings.embed_query(target_dossier)
    doc_embs  = rag.embeddings.embed_documents([d.page_content for d in all_docs[:DIVERSITY_K]])
    sims_q    = cosine_similarity([query_emb], doc_embs)[0]
    TH = 0.75
    ctx_prec = sum(1 for s in sims_q if s>=TH)/len(sims_q) if sims_q.size else 0

    # p95 latency
    p95 = round(1000*sorted(latencies)[int(0.95*len(latencies))],1)

    records.append({
        "chunk_size":      ch_size,
        "chunk_overlap":   ch_overlap,
        "k_retriever":     k_retriever,
        "mmr_lambda":      mmr_lambda,
        "temperature":     temperature,
        "num_chunks":      n_chunks,
        "indexing_sec":    round(idx_time,2),
        "throughput_cps":  round(throughput,1),
        "mem_before_idx":  round(mem_before_idx,1),
        "mem_after_idx":   round(mem_after_idx,1),
        "mem_before_q":    round(mem_before_q,1),
        "mem_after_q":     round(mem_after_q,1),
        "p95_latency_ms":  p95,
        "redundancy":      round(redundancy,3),
        "diversity":       round(diversity,3),
        "faithfulness":    round(faith_score,3),
        "context_precision": round(ctx_prec,3),
    })

df = pd.DataFrame(records)
df.to_csv("chunk_bench_grid_search.csv", index=False)
print("✓ Done → chunk_bench_grid_search.csv")


