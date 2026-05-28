# External knowledge sources

Five permissively-licensed documentation corpora that broaden the alignment
training set beyond Portll's own code. Each source is fetched into
`pairs/_external/<repo>/` (gitignored) via `fetch.ps1` (Windows) or
`fetch.sh` (POSIX), then consumed by `alignment-collect` through the TOML
specs alongside this README.

The selection targets the languages the system most often answers questions
about during tool use: JavaScript/TypeScript/React, Python, Rust. PostgreSQL
is deferred because its docs ship as SGML which would need an extractor.

## Sources

| TOML spec                       | Repo                                     | License       | Subtree fetched                                   | Format    |
|---------------------------------|------------------------------------------|---------------|---------------------------------------------------|-----------|
| `mdn-javascript.toml`           | mdn/content                              | CC-BY-SA-2.5  | `files/en-us/web/javascript/`                     | Markdown  |
| `python-tutorial.toml`          | python/cpython                           | PSF-2.0       | `Doc/tutorial/`, `Doc/library/`                   | reST      |
| `rust-by-example.toml`          | rust-lang/rust-by-example                | MIT/Apache-2.0| `src/`                                            | Markdown  |
| `react-docs.toml`               | reactjs/react.dev                        | CC-BY-4.0     | `src/content/learn/`, `src/content/reference/`    | MDX       |
| `typescript-handbook.toml`      | microsoft/TypeScript-Website             | CC-BY-4.0     | `packages/documentation/copy/en/`                 | Markdown  |

All five formats are text-y enough that the existing paragraph chunker in
`alignment-collect` handles them without modification. reST / MDX
directives appear as inline literals; the embedder treats them as content.

## How to fetch

```powershell
# Windows
./fetch.ps1
```

```bash
# POSIX
./fetch.sh
```

Both scripts use `git clone --depth=1 --filter=blob:none --sparse` so each
checkout is small (typically <20MB per source after sparse subtree
selection). They write to `evaluations/alignment/pairs/_external/<repo>/`,
which is gitignored.

## How to include in a fit

```powershell
./target/release/alignment-collect.exe `
  --sources evaluations/alignment/pairs/sources/veld-rust.toml `
  --sources evaluations/alignment/pairs/sources/external/mdn-javascript.toml `
  --sources evaluations/alignment/pairs/sources/external/python-tutorial.toml `
  --sources evaluations/alignment/pairs/sources/external/rust-by-example.toml `
  --sources evaluations/alignment/pairs/sources/external/react-docs.toml `
  --sources evaluations/alignment/pairs/sources/external/typescript-handbook.toml `
  --out evaluations/alignment/pairs/pairs.jsonl `
  --quota 5000
```

The per-domain quota (5000 by default) caps each domain bucket. Sources
that share a domain (e.g. `programming` covers Python + Rust + the existing
veld+sleight repos) compete for that bucket on a first-come basis. To get
diverse representation, list domain-thin sources first.

## Domain mapping

| Source             | Domain            |
|--------------------|-------------------|
| mdn-javascript     | web_development   |
| react-docs         | web_development   |
| typescript-handbook| web_development   |
| python-tutorial    | programming       |
| rust-by-example    | programming       |

The existing internal sources (`veld-rust.toml`, `veld-md.toml`, etc.)
already populate `programming` and `docs`. The external set adds
`web_development` and broadens `programming`.

## License audit

Each external source's license is recorded in the row's `license` field. A
final license audit on the merged `pairs.jsonl` is straightforward:

```powershell
Get-Content evaluations/alignment/pairs/pairs.jsonl |
  ForEach-Object { (ConvertFrom-Json $_).license } |
  Group-Object | Select-Object Count, Name
```

Any non-permissive license appearing in the output is a contribution-policy
violation per `evaluations/alignment/CONTRIBUTING.md`.
