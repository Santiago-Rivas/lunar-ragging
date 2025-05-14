#!/usr/bin/env python3
"""
remove_empty.py
Delete empty celebrity-<id>.md files and remove their rows from users.csv
"""

import os
import re
import shutil
import pandas as pd
from pathlib import Path

# ---------- CONFIG -------------------------------------------------------------
CSV_NAME   = "users.csv"          # name of the CSV file
BACKUP_EXT = ".bak"               # extension added to the CSV backup
ID_COL     = "id"                 # column in the CSV that holds the numeric ID
PATTERN    = re.compile(r"^celebrity-(\d+)\.md$", re.IGNORECASE)
# ------------------------------------------------------------------------------

def is_really_empty(path: Path) -> bool:
    """True if the file has 0 bytes **or** contains only whitespace."""
    if path.stat().st_size == 0:
        return True
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip() == ""

def main() -> None:
    here      = Path.cwd()
    csv_path  = here / CSV_NAME
    df        = pd.read_csv(csv_path, sep=";")

    removed_ids = []

    # Walk every .md file in the folder
    for md_file in here.glob("*.md"):
        m = PATTERN.match(md_file.name)
        if not m:
            continue                          # skip non-conforming names

        file_id = int(m.group(1).lstrip("0") or "0")  # “0007” → 7, “0” → 0

        if is_really_empty(md_file):
            md_file.unlink()
            removed_ids.append(file_id)
            print(f"Removed empty {md_file}")

    if removed_ids:
        # Backup CSV
        backup = csv_path.with_suffix(csv_path.suffix + BACKUP_EXT)
        shutil.copy2(csv_path, backup)
        print(f"CSV backed up to {backup}")

        # Remove rows whose id is in removed_ids
        df = df[~df[ID_COL].astype(int).isin(removed_ids)]
        df.to_csv(csv_path, sep=";", index=False)
        print(f"Updated CSV saved to {csv_path}")
    else:
        print("No empty files found — nothing changed.")

if __name__ == "__main__":
    main()
