# Contributing alignment training data

The veld alignment corpus is **code and public sources only**. Every row must
be redistributable under a permissive license. We refit the global alignment
on a fixed cadence; contributions feed that fit and ship to every downstream
veld install.

## What we accept

- Source code under MIT, Apache-2.0, BSD-3-Clause, MPL-2.0, or compatible.
- Public documentation, READMEs, RFCs, ADRs, changelog entries under CC-BY,
  CC0, or the source repository's own permissive license.
- Stack Overflow content under CC-BY-SA-4.0 (attribution retained in `source`).
- Synthetic prompts/completions you authored yourself.

## What we refuse — no exceptions

- **Personal correspondence**: email, chat, DMs, Slack/Discord/Teams exports.
- **Document stores**: anything pulled from a private Google Drive, Notion,
  Confluence, OneDrive, or similar.
- **Robotics and sensor telemetry**: Zenoh/ROS2 topic captures, IMU traces,
  GPS coordinates, camera frames, audio clips. Even when "anonymized," device
  IDs and timestamps re-identify trivially.
- **Biometric data**: voice samples, gait sequences, gaze tracking, anything
  derived from a person's body.
- **PII of any kind**: names, addresses, phone numbers, IPs in logs, tokens,
  API keys, account numbers, license plates, faces.
- **Proprietary corporate IP**: code from a closed-source repository, customer
  data, internal-only specifications, trade secrets.
- **Anything you would not want indexed by a public search engine.**

If a row would embarrass you, your employer, or your subject, it doesn't belong
here.

## Submission format

One JSONL file, UTF-8, one object per line:

```json
{
  "text": "<the text>",
  "domain": "<one of the listed domains>",
  "license": "<SPDX identifier>",
  "source": "<short label that identifies the origin>"
}
```

Accepted domains: `web_development`, `project_management`, `database`,
`programming`, `analytics`, `devops`, `docs`, `security`, `testing`,
`ai_loop`.

License field must be a single SPDX identifier (`MIT`, `Apache-2.0`,
`BSD-3-Clause`, `CC-BY-SA-4.0`, `CC0-1.0`, `BUSL-1.1`, etc.). Rows with
non-SPDX or non-permissive licenses are dropped during ingest.

## Size and shape

- Each row 64–2048 characters of text.
- 500–5000 rows per contribution is the sweet spot.
- Diverse within a domain — don't submit 5000 rows from one file.
- De-duplicate by content on your end before submission.

## Process

1. Open a PR adding a TOML source spec at
   `evaluations/alignment/pairs/sources/friends/<your-handle>.toml`.
2. The JSONL file itself is gitignored by default. Commit it only if you've
   confirmed the license permits redistribution. Otherwise keep it local and
   reference it from the spec.
3. CI runs `alignment-fit` against the augmented corpus. If the held-out
   cosine improves (or holds within margin), the PR merges.

## Audit and revocation

Every published alignment ships with a manifest of contributing source labels.
If a contribution is later determined to violate this policy, we revoke the
alignment, refit without the offending source, and publish a new version. The
content itself is removed from the corpus.

## Code of conduct

Don't contribute anything you wouldn't want indexed by a public search engine.
If you're not sure whether a corpus is permitted, ask before opening a PR.
