import csv, json, pathlib, re

# ---------- 0.  PATHS ----------
prompt_path   = pathlib.Path("input/dossier_generation_prompt.md")
event_json    = pathlib.Path("input/space_event.json")
input_csv     = pathlib.Path("input/celeb_merged.csv")
output_jsonl  = pathlib.Path("input/dossier_batchinput.jsonl")

# ---------- 1.  LOAD PROMPT TEMPLATE ----------
if not prompt_path.is_file():
    raise FileNotFoundError(f"System-prompt file not found: {prompt_path}")
template_prompt = prompt_path.read_text(encoding="utf-8")

# ---------- 2.  LOAD EVENT-LEVEL VARIABLES ----------
if not event_json.is_file():
    raise FileNotFoundError(f"Event-variable file not found: {event_json}")
event_vars = json.loads(event_json.read_text(encoding="utf-8"))

# Ensure strings (helps with .replace)
event_vars = {k: str(v) for k, v in event_vars.items()}

# ---------- 3.  HELPER: SUBSTITUTE {{TOKENS}} ----------
token_re = re.compile(r"\{\{(\w+?)\}\}")

def fill_placeholders(text: str, extra_vars: dict) -> str:
    """Replace all {{TOKEN}} in text with values from event_vars ∪ extra_vars."""
    mapping = {**event_vars, **extra_vars}
    def _sub(match):
        key = match.group(1)
        return mapping.get(key, match.group(0))   # leave token intact if not found
    return token_re.sub(_sub, text)

# ---------- 4.  BUILD THE BATCH JSONL ----------
with input_csv.open(newline="", encoding="utf-8") as infile, \
     output_jsonl.open("w",           encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter=";")
    for row in reader:
        celeb_id   = row["id"].strip()
        celeb_name = row["name"].strip()

        # Per-request variables
        per_req_vars = {
            "INDIVIDUAL_NAME": celeb_name,
        }

        system_prompt_filled = fill_placeholders(template_prompt, per_req_vars)

        batch_line = {
            "custom_id": f"celebrity-{celeb_id.zfill(4)}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "o3",
                "messages": [
                    {"role": "system", "content": system_prompt_filled}
                ],
                "max_tokens": 15000
            }
        }
        outfile.write(json.dumps(batch_line, ensure_ascii=False) + "\n")

print(f"✅  JSONL written to: {output_jsonl}")
