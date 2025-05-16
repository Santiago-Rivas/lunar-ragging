import time, psutil
import csv
import os
import sys
from pathlib import Path
from itertools import product, combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
import openai
import shutil
import threading

# Add parent directory to path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

from ragas import evaluate
from ragas.metrics import Faithfulness
from tests.rag_refactored import ConnectionRAG, ingest_from_dirs

DATA_DOSSIER = Path(__file__).parent.parent / "data/space_data/dossiers"
DATA_HTML    = Path(__file__).parent.parent / "data/space_data/html"
EVENT_ID     = "F"
OUTPUT_FILE  = Path(__file__).parent / f"chunk_bench_parallel_grid_search.csv"
WORKERS      = 5
MAX_USERS    = 2
# Maximum number of concurrent users to process per parameter combination
MAX_USER_THREADS = 10

DIVERSITY_K   = 10
faith_metric  = Faithfulness()

# grid of parameters to sweep
# chunk_sizes    = [512, 1024, 2048, 4096]
# chunk_overlaps = [64, 128, 256]
# k_retriever_ls = [5, 10, 20]
# mmr_lambdas    = [0.0, 0.3, 0.5, 0.7]
# temperatures   = [0.0, 0.2, 0.5]

chunk_sizes    = [1024]
chunk_overlaps = [128]
k_retriever_ls = [20]
mmr_lambdas    = [0.8]
temperatures   = [0.7]

# Lock for thread-safe CSV writing
csv_lock = threading.Lock()

def load_user_ids():
    """Load all user IDs from the users.csv file"""
    users_file = DATA_DOSSIER / "users.csv"
    if not users_file.exists():
        print(f"Warning: {users_file} not found. Falling back to default TARGET_ID='1'")
        return ["1"]
    
    user_ids = []
    with open(users_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            user_ids.append(row['id'])
    
    print(f"Loaded {len(user_ids)} user IDs from {users_file}")
    return user_ids

def write_record(record, first_write=False):
    """Write a single record to the CSV file"""
    with csv_lock:
        mode = 'w' if first_write else 'a'
        with open(str(OUTPUT_FILE), mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if first_write:
                writer.writeheader()
            writer.writerow(record)

def get_index_key(ch_size, ch_overlap):
    """Generate a unique key for a given chunk size and overlap combination"""
    return f"{ch_size}_{ch_overlap}"

def process_single_target(target_id, rag, params, n_chunks, idx_time, mem_before_idx, mem_after_idx):
    """Process a single target user and return the result"""
    ch_size, ch_overlap, k_retriever, mmr_lambda, temperature = params
    proc = psutil.Process()
    
    print(f"Processing target_id={target_id}")
    # retriever with MMR params
    where = {"$and":[{"event_id":{"$eq":EVENT_ID}},{"user_id":{"$ne":target_id}}]}
    retriever = rag.collection.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           k_retriever,
            "filter":      where,
            "lambda_mult": mmr_lambda
        },
    )

    # fetch dossier
    hit = rag.collection.get(where={"user_id":{"$eq":target_id}},
                             limit=1, include=["metadatas"])
    target_name    = hit["metadatas"][0].get("name","")    if hit["metadatas"] else ""
    target_dossier = hit["metadatas"][0].get("dossier","") if hit["metadatas"] else ""

    # querying & collect
    mem_before_q = proc.memory_info().rss / 1024**2
    latencies, raw_answer, all_docs = [], None, None
    docs = retriever.invoke(target_dossier)
    all_docs = docs

    parts = [{"user_id":d.metadata["user_id"],
              "name":   d.metadata.get("name",""),
              "chunk":  d.page_content}
             for d in docs[:10]]
    prompt = rag.build_prompt(target_id, target_name, target_dossier, parts, k=5)

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
        "question": [prompt],
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

    record = {
        "target_id":       target_id,
        "chunk_size":      ch_size,
        "chunk_overlap":   ch_overlap,
        "k_retriever":     k_retriever,
        "mmr_lambda":      mmr_lambda,
        "temperature":     temperature,
        "num_chunks":      n_chunks,
        "indexing_sec":    round(idx_time,2),
        "mem_before_idx":  round(mem_before_idx,1),
        "mem_after_idx":   round(mem_after_idx,1),
        "mem_before_q":    round(mem_before_q,1),
        "mem_after_q":     round(mem_after_q,1),
        "redundancy":      round(redundancy,3),
        "diversity":       round(diversity,3),
        "faithfulness":    round(faith_score,3),
        "context_precision": round(ctx_prec,3),
    }
    
    # Write the record immediately after test completion
    first_write = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    write_record(record, first_write)
    
    print(f"Completed test for target_id={target_id} with chunk_size={ch_size}, chunk_overlap={ch_overlap}")
    
    # Clean up target-specific resources
    del retriever, all_docs, sample, ds, df_eval, query_emb, doc_embs, sims_q
    if 'embs' in locals(): del embs
    if 'sims' in locals(): del sims
    
    return 1  # Return 1 to count the completion

def run_test_for_config(params, target_ids):
    """Run tests with given parameters for multiple target_ids, reusing the same database instance"""
    ch_size, ch_overlap, k_retriever, mmr_lambda, temperature = params
    proc = psutil.Process()  # Create a new process object in each worker
    
    # Create a process-specific cache directory for this worker
    pid = os.getpid()
    index_key = get_index_key(ch_size, ch_overlap)
    cache_base = Path(__file__).parent / "cache"
    cache_base.mkdir(exist_ok=True)
    cache_dir = cache_base / f"cache_{pid}_{index_key}"
    
    # init RAG with custom splitter - this will be reused for all targets
    rag = ConnectionRAG(persist_directory=str(cache_dir), 
                        chunk_size=ch_size, 
                        chunk_overlap=ch_overlap)

    # Check if this process has already indexed this configuration
    if not cache_dir.exists() or not list(cache_dir.glob("*.bin")):
        # Need to perform indexing
        mem_before_idx = proc.memory_info().rss / 1024**2
        t0 = time.perf_counter()
        ingest_from_dirs(rag, DATA_DOSSIER, DATA_HTML, EVENT_ID)
        idx_time = time.perf_counter() - t0
        mem_after_idx = proc.memory_info().rss / 1024**2
        print(f"Process {pid}: Indexed data for chunk_size={ch_size}, chunk_overlap={ch_overlap}")
    else:
        # Already indexed, use existing data
        print(f"Process {pid}: Using cached index for chunk_size={ch_size}, chunk_overlap={ch_overlap}")
        mem_before_idx = 0
        mem_after_idx = 0  
        idx_time = 0
    
    # throughput
    n_chunks = len(rag.collection.get(include=[])["ids"])
    throughput = n_chunks / idx_time if idx_time>0 else 0
    
    results = 0
    
    # Process target users in parallel using ThreadPoolExecutor
    max_workers = min(MAX_USER_THREADS, len(target_ids))
    print(f"Processing {len(target_ids)} target users with {max_workers} threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with the shared parameters
        process_func = partial(
            process_single_target, 
            rag=rag, 
            params=params,
            n_chunks=n_chunks,
            idx_time=idx_time,
            mem_before_idx=mem_before_idx,
            mem_after_idx=mem_after_idx
        )
        
        # Submit all target IDs to the thread pool
        futures = [executor.submit(process_func, target_id) for target_id in target_ids]
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                results += future.result()
            except Exception as e:
                print(f"Error processing target: {e}")
    
    # Only delete the RAG system after all targets have been processed
    del rag
    
    return results

def generate_averaged_results():
    """Generate a summary CSV with averaged metrics across all target IDs"""
    df = pd.read_csv(OUTPUT_FILE)
    
    # Group by parameter combinations and calculate means
    param_cols = ['chunk_size', 'chunk_overlap', 'k_retriever', 'mmr_lambda', 'temperature']
    metric_cols = [col for col in df.columns if col not in param_cols and col != 'target_id']
    
    avg_df = df.groupby(param_cols)[metric_cols].mean().reset_index()
    
    # Add count of targets tested for each parameter combination
    targets_per_param = df.groupby(param_cols)['target_id'].nunique().reset_index()
    targets_per_param.rename(columns={'target_id': 'targets_tested'}, inplace=True)
    avg_df = pd.merge(avg_df, targets_per_param, on=param_cols)
    
    # Round numeric columns for readability
    for col in metric_cols:
        if avg_df[col].dtype in [np.float64, np.float32]:
            avg_df[col] = avg_df[col].round(3)
    
    avg_file = Path(__file__).parent / "chunk_bench_averaged_results.csv"
    avg_df.to_csv(avg_file, index=False)
    print(f"✓ Done → {avg_file}")

def main():
    # Get all user IDs to test
    user_ids = load_user_ids()
    
    # Limit number of users if MAX_USERS is set
    if MAX_USERS is not None and MAX_USERS > 0 and len(user_ids) > MAX_USERS:
        print(f"Limiting to {MAX_USERS} users (from {len(user_ids)} total)")
        user_ids = user_ids[:MAX_USERS]
    
    print(f"Testing with {len(user_ids)} target users: {', '.join(user_ids[:5])}" + 
          ("..." if len(user_ids) > 5 else ""))
    
    # Clean up any existing cache directories from previous runs
    cache_base = Path(__file__).parent / "cache"
    if cache_base.exists():
        for cache_dir in cache_base.glob("cache_*"):
            if cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir)
                    print(f"Removed old cache: {cache_dir}")
                except Exception as e:
                    print(f"Failed to remove {cache_dir}: {e}")
    
    # Generate all parameter combinations
    param_combinations = list(product(
        chunk_sizes, chunk_overlaps, k_retriever_ls, mmr_lambdas, temperatures
    ))
    
    total_tasks = len(param_combinations)
    total_tests = len(param_combinations) * len(user_ids)
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Total test cases: {total_tests}")
    print(f"Using {WORKERS} worker processes")
    
    completed_params = 0
    completed_tests = 0
    first_write = True
    
    # Run tests in parallel with one task per parameter combination (reusing DB across targets)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # Submit one task per parameter combination
        futures = [executor.submit(run_test_for_config, params, user_ids) for params in param_combinations]
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                records = future.result()
                
                # Records are already written to the CSV in the worker process
                # Just update the progress counters
                completed_params += 1
                completed_tests += records
                print(f"Progress: {completed_tests}/{total_tests} ({completed_tests/total_tests*100:.1f}%) completed")
                print(f"Completed parameter combination {completed_params}/{total_tasks} "
                      f"({completed_params/total_tasks*100:.1f}%)")
            except Exception as e:
                print(f"Error in task: {e}")
    
    print(f"✓ Done → {OUTPUT_FILE}")
    
    # Generate averaged results
    generate_averaged_results()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up cache directories
        cache_base = Path(__file__).parent / "cache"
        if cache_base.exists():
            for cache_dir in cache_base.glob("cache_*"):
                if cache_dir.is_dir():
                    try:
                        shutil.rmtree(cache_dir)
                        print(f"Cleaned up cache: {cache_dir}")
                    except Exception as e:
                        print(f"Failed to clean up {cache_dir}: {e}") 