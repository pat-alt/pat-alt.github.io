#!/usr/bin/env bash
# Render all CV PDFs locally.
#
# Requires:
#   - papis, with the pat-alt/bib library configured (e.g. via its setup.sh)
#   - quarto with a TeX engine (lualatex)
#
# Exports the split bibliographies from the papis library by tag and renders
# every CV entry point in www/ to PDF. resume.qmd and resume_deutsch.qmd are
# included into the entry points, so they are not rendered standalone.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

for cmd in papis quarto; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: $cmd not found on PATH" >&2
    exit 1
  fi
done

mkdir -p bib

# Split bibliographies, one per CV topic. The keys must match the
# `bibliography:` map in www/full.qmd and www/academic.qmd.
papis export --all 'tags:peerreviewed'   --batch --format bibtex --out bib/peer.bib
papis export --all 'tags:other'          --batch --format bibtex --out bib/other.bib
papis export --all 'tags:preprint'       --batch --format bibtex --out bib/pre.bib
papis export --all 'tags:sup-masters'    --batch --format bibtex --out bib/sup_masters.bib
papis export --all 'tags:sup-bachelors'  --batch --format bibtex --out bib/sup_bachelors.bib
papis export --all 'tags:software'       --batch --format bibtex --out bib/soft.bib

# Render every CV entry point, one at a time.
for cv in www/full.qmd www/academic.qmd www/industry.qmd www/industry_deutsch.qmd; do
  quarto render "$cv"
done