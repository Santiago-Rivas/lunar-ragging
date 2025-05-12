"""
generate_dossiers_async.py  – v1-SDK compatible
-----------------------------------------------
> pip install "openai>=1.3.0" aiofiles tenacity tqdm
"""

import asyncio, csv, json, pathlib, re, os
import aiofiles
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm.asyncio import tqdm_asyncio

print(__doc__, flush=True)        # -> writes the doc-string to stdout

load_dotenv()
# -------- configuration --------
prompt_path   = pathlib.Path("input/dossier_generation_prompt.md")
event_json    = pathlib.Path("input/space_event.json")
input_csv     = pathlib.Path("input/celeb_merged.csv")
out_dir       = pathlib.Path("output/dossiers")
log_path      = pathlib.Path("output/dossier_log.csv")

PARALLEL_REQUESTS = 10
MODEL_NAME        = "o3"
MAX_TOKENS        = 1500

# -------- load prompt & event vars --------
template_prompt = prompt_path.read_text(encoding="utf-8")
event_vars = {k: str(v) for k, v in json.loads(event_json.read_text("utf-8")).items()}

token_re = re.compile(r"\{\{(\w+?)\}\}")
fill_ph  = lambda t, extra: token_re.sub(lambda m: {**event_vars, **extra}.get(m.group(1), m.group(0)), t)

# -------- create a single async client --------
client = AsyncOpenAI()      # uses env var OPENAI_API_KEY

# -------- retry-wrapped request --------
@retry(wait=wait_random_exponential(2, 20), stop=stop_after_attempt(6))
async def get_completion(prompt: str) -> str:
    resp = await client.chat.completions.create(
        model      = MODEL_NAME,
        messages   = [{"role": "system", "content": prompt}],
        max_completion_tokens = MAX_TOKENS,
    )
    return resp.choices[0].message.content

# -------- per-row task --------
async def process(row: dict, sem: asyncio.Semaphore):
    celeb_id, celeb_name = row["id"].strip(), row["name"].strip()
    cid   = f"celebrity-{celeb_id.zfill(4)}"
    prompt = fill_ph(template_prompt, {"INDIVIDUAL_NAME": celeb_name})

    try:
        async with sem:
            dossier = await get_completion(prompt)
        out_file = out_dir / f"{cid}.md"
        async with aiofiles.open(out_file, "w", encoding="utf-8") as f:
            await f.write(dossier)
        return cid, True, "ok"
    except Exception as e:
        print(e)
        return cid, False, str(e)

# -------- orchestrator --------
async def main():
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(input_csv.open(encoding="utf-8"), delimiter=";"))
    # rows = rows[:10]
    sem  = asyncio.Semaphore(PARALLEL_REQUESTS)

    results = await tqdm_asyncio.gather(*(process(r, sem) for r in rows),
                                        desc="Generating dossiers")

    # save log CSV
    with log_path.open("w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf); writer.writerow(["custom_id","success","info"]); writer.writerows(results)

    ok = sum(1 for _,s,_ in results if s)
    print(f"✅  Finished {ok}/{len(results)} dossiers.  Log: {log_path}")

if __name__ == "__main__":
    asyncio.run(main())
