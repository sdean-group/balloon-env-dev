#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

command -v pdflatex >/dev/null || {
  echo "pdflatex is required" >&2
  exit 1
}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3" ]]; then
  PYTHON="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
else
  PYTHON="python3"
fi
"$PYTHON" -c 'import pypdf, reportlab' >/dev/null || {
  echo "Python packages pypdf and reportlab are required" >&2
  exit 1
}

pdflatex -interaction=nonstopmode -halt-on-error -jobname=poster_overlay \
  '\def\overlayonly{1}\input{poster.tex}'
"$PYTHON" merge_overlay.py
"$PYTHON" flatten_poster.py

echo "Built $ROOT/poster_vector.pdf (editable)"
echo "Built $ROOT/poster.pdf (flattened delivery copy)"
