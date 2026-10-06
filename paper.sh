#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
paper_dir="$repo_dir/paper"

if [[ ! -d "$paper_dir" ]]; then
    printf 'paper directory not found: %s\n' "$paper_dir" >&2
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    printf 'zip command is required; install it and try again.\n' >&2
    exit 1
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
serial=1
while :; do
    printf -v suffix '%03d' "$serial"
    archive="$repo_dir/paper_${timestamp}_${suffix}.zip"
    [[ ! -e "$archive" ]] && break
    serial=$((serial + 1))
done

# Run from paper/ so main.tex and the other project files land at the ZIP root,
# which lets Overleaf find the main document after upload.
(
    cd "$paper_dir"
    zip -r -X "$archive" . \
        -x './.gitignore' './.DS_Store' '*/.DS_Store' \
        '*.aux' '*.bak' '*.bbl' '*.blg' '*.brf' '*.log' '*.out' \
        '*.fls' '*.synctex.gz' '*.fdb_latexmk' '*.pdf'
)

printf 'Created Overleaf archive: %s\n' "$archive"
