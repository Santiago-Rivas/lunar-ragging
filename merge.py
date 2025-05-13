import pandas as pd, pathlib, re

# -------- 1)  paths to the two CSVs you just posted --------
csv_400_path = pathlib.Path("input/celeb_400.csv")   # first list (400 rows)
csv_182_path = pathlib.Path("input/space_mariano.csv")   # second list (182 rows)

# -------- 2)  read them --------
df_400 = pd.read_csv(csv_400_path, delimiter=";")
df_182 = pd.read_csv(csv_182_path, delimiter=";")

# -------- 3)  helper to normalise names --------
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

df_400["name_clean"] = df_400["name"].apply(clean)
df_182["name_clean"] = df_182["name"].apply(clean)

# -------- 4)  concat + dedupe --------
merged = (
    pd.concat([df_400, df_182], ignore_index=True)
      .drop_duplicates(subset="name_clean", keep="first")
      .drop(columns="name_clean")
      .reset_index(drop=True)
)

# -------- 5)  renumber ids --------
merged["id"] = range(1, len(merged) + 1)

# -------- 6)  save --------
out_path = pathlib.Path("input/celeb_merged.csv")
merged.to_csv(out_path, sep=";", index=False)

print(f"✅  Merged CSV saved to {out_path} — {len(merged)} unique entries.")

