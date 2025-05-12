#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/dossiers
csv="data/dossiers/id_to_name.csv"
printf 'id;name\n' > "$csv"               # header + newline

i=1
#            ▼ raw-string output — no quotes around the Base64 blob
jq -r '.[] | @base64' dossiers.json |
while IFS= read -r row; do
  _jq() { echo "$row" | base64 --decode | jq -r "$1"; }

  name=$(_jq '.name')                     # e.g. "Lionel Messi"
  text=$(_jq '.dossier')                  # markdown body

  printf '%s\n' "$text" > "data/dossiers/${i}.md"
  printf '%s;%s\n' "$i" "$name" >> "$csv"

  echo "wrote dossier ${i}.md  (name: $name)"
  ((i++))
done
