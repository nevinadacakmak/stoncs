#!/usr/bin/env zsh
# Load .env into the current shell session safely.
# Usage: source scripts/load_env.sh
# This script intentionally uses a safe parser (doesn't `eval` the file).

ENVFILE=.env
if [ ! -f "$ENVFILE" ]; then
  echo ".env not found in $(pwd)"
  return 1
fi

while IFS= read -r line || [ -n "$line" ]; do
  # Strip leading/trailing whitespace
  line="${line#${line%%[![:space:]]*}}"
  line="${line%${line##*[![:space:]]}}"
  # Skip empty lines and comments
  [[ -z "$line" || "$line" = \#* ]] && continue
  # Skip lines without '='
  [[ "$line" != *=* ]] && continue
  key=${line%%=*}
  value=${line#*=}
  # Remove surrounding quotes from value if present
  if [[ "$value" = '"'*'"' || "$value" = "'"*"'" ]]; then
    value=${value:1:-1}
  fi
  export "$key"="$value"
done < "$ENVFILE"

echo ".env loaded into current shell (use this terminal to run Streamlit or scripts)."
