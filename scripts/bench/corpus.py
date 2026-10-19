"""Deterministic benchmark corpus: one JSON object per line, identical for a given count and seed."""
import argparse
import json
import random
import sys

SUBJECTS = ["the ingest worker", "the billing service", "a deploy pipeline", "the search index",
            "our auth gateway", "the nightly batch", "a cache layer", "the mobile client",
            "the scheduler", "a vector store", "the audit log", "the export job"]
VERBS = ["was reconfigured after", "regressed following", "stabilised once we changed",
         "stalled because of", "recovered when we rolled back", "was profiled during",
         "needed a retry policy for", "was documented alongside"]
OBJECTS = ["a connection pool limit", "an expired credential", "a schema migration",
           "a noisy neighbour on the host", "a stale lock file", "a partial network outage",
           "an oversized payload", "a clock skew between nodes", "a missing index",
           "a misconfigured timeout"]
FILLER = ["The change was reviewed by two engineers and merged after the integration suite passed.",
          "Latency returned to baseline within a few minutes and no data was lost.",
          "A follow-up ticket tracks the remaining cleanup and the alert threshold.",
          "The incident notes record the exact sequence of commands that were run.",
          "Capacity headroom was checked against the previous quarter before the rollout.",
          "The owning team agreed to revisit the decision after the next release cycle."]
TAGS = ["infra", "billing", "search", "auth", "mobile", "batch", "cache", "audit", "deploy", "incident"]


def build_record(index, rng):
    sentence = "Item %d: %s %s %s." % (
        index, rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS))
    body = [sentence]
    while len(" ".join(body)) < 280:
        body.append(rng.choice(FILLER))
    return {
        "content": " ".join(body)[:320],
        "tags": rng.sample(TAGS, 2),
        "episode_id": "episode-%d" % (index // 50),
        "sequence_number": index % 50,
    }


def main():
    parser = argparse.ArgumentParser(description="Write a deterministic benchmark corpus as JSON lines.")
    parser.add_argument("--count", type=int, required=True, help="number of records")
    parser.add_argument("--seed", type=int, default=20261018, help="RNG seed (default 20261018)")
    parser.add_argument("--out", required=True, help="output file path")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        for index in range(args.count):
            handle.write(json.dumps(build_record(index, rng), separators=(",", ":")) + "\n")
    print("wrote %d records to %s" % (args.count, args.out), file=sys.stderr)


if __name__ == "__main__":
    main()
