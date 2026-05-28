#!/usr/bin/env bash
# Fetch external knowledge sources for alignment corpus expansion.
#
# Each source is a sparse, shallow git clone of just the doc subtree —
# typically <20 MB per repo. Re-running the script updates existing clones
# rather than re-fetching from scratch.
#
# Writes to `evaluations/alignment/pairs/_external/` (gitignored).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
external_dir="${repo_root}/evaluations/alignment/pairs/_external"

mkdir -p "${external_dir}"

# folder | repo url | subtree
sources=(
    # Original five: programming + web_development + docs
    "mdn-content|https://github.com/mdn/content.git|files/en-us/web/javascript"
    "cpython|https://github.com/python/cpython.git|Doc"
    "rust-by-example|https://github.com/rust-lang/rust-by-example.git|src"
    "react.dev|https://github.com/reactjs/react.dev.git|src/content"
    "TypeScript-Website|https://github.com/microsoft/TypeScript-Website.git|packages/documentation"
    # Seven new: project_management / database / analytics / devops / security / testing / ai_loop
    "rust-rfcs|https://github.com/rust-lang/rfcs.git|text"
    "duckdb-web|https://github.com/duckdb/duckdb-web.git|docs"
    "spark|https://github.com/apache/spark.git|docs"
    "kubernetes-website|https://github.com/kubernetes/website.git|content/en/docs/concepts"
    "OWASP-CheatSheetSeries|https://github.com/OWASP/CheatSheetSeries.git|cheatsheets"
    "pytest|https://github.com/pytest-dev/pytest.git|doc/en"
    "transformers|https://github.com/huggingface/transformers.git|docs/source/en"
)

for entry in "${sources[@]}"; do
    IFS='|' read -r folder repo sub <<< "${entry}"
    dest="${external_dir}/${folder}"
    if [[ -d "${dest}" ]]; then
        echo "[fetch] ${folder} exists — pulling updates"
        (cd "${dest}" && git pull --depth=1 origin HEAD)
    else
        echo "[fetch] cloning ${repo} (sparse: ${sub})"
        git clone --depth=1 --filter=blob:none --sparse "${repo}" "${dest}"
        (cd "${dest}" && git sparse-checkout set "${sub}")
    fi
done

echo ""
echo "[fetch] done. Sources available under ${external_dir}"
echo "[fetch] next: run alignment-collect with the TOML specs in this directory."
