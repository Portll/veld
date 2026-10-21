"""Assemble per-engine result JSON and render the side-by-side markdown table."""
import argparse
import glob
import json
import os
import sys

PHASE_FILES = ("remember", "recall_cold", "recall_warm", "proactive")


def assemble(args):
    phases = {}
    for name in PHASE_FILES:
        with open(os.path.join(args.phases_dir, name + ".json"), "r", encoding="utf-8") as handle:
            phases[name] = json.load(handle)
    document = {
        "engine": args.engine,
        "memories": args.memories,
        "concurrency": args.concurrency,
        "tuning_env": dict(item.split("=", 1) for item in args.tuning if "=" in item),
        "phases": phases,
        "rss_after_seed_kb": args.rss_seed_kb,
        "rss_after_recall_kb": args.rss_recall_kb,
        "data_dir_bytes": args.data_bytes,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def fmt_rate(phase):
    return "n/a" if phase.get("req_per_s") is None else "%.1f" % phase["req_per_s"]


def fmt_p95(phase):
    return "n/a" if phase.get("p95_ms") is None else "%.1f ms" % phase["p95_ms"]


def fmt_mb(kilobytes):
    return "n/a" if kilobytes is None else "%.0f MB" % (kilobytes / 1024.0)


def fmt_data(size_bytes):
    return "n/a" if size_bytes is None else "%.1f MB" % (size_bytes / 1048576.0)


def table(args):
    documents = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            documents.append(json.load(handle))
    if not documents:
        sys.exit("no result files in " + args.results_dir)
    rows = [
        ("remember req/s", lambda d: fmt_rate(d["phases"]["remember"])),
        ("remember p95", lambda d: fmt_p95(d["phases"]["remember"])),
        ("recall cold req/s", lambda d: fmt_rate(d["phases"]["recall_cold"])),
        ("recall warm req/s", lambda d: fmt_rate(d["phases"]["recall_warm"])),
        ("recall p95 warm", lambda d: fmt_p95(d["phases"]["recall_warm"])),
        ("proactive_context req/s", lambda d: fmt_rate(d["phases"]["proactive"])),
        ("RSS after seed", lambda d: fmt_mb(d["rss_after_seed_kb"])),
        ("RSS after recall", lambda d: fmt_mb(d["rss_after_recall_kb"])),
        ("data dir size", lambda d: fmt_data(d["data_dir_bytes"])),
        ("request errors", lambda d: str(sum(p["errors"] for p in d["phases"].values()))),
    ]
    print("| metric (%d memories, concurrency %d) | %s |" % (
        documents[0]["memories"], documents[0]["concurrency"], " | ".join(d["engine"] for d in documents)))
    print("|---|" + "---|" * len(documents))
    for label, render in rows:
        print("| %s | %s |" % (label, " | ".join(render(d) for d in documents)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble", help="merge phase files and measurements into one engine result")
    a.add_argument("--engine", required=True)
    a.add_argument("--phases-dir", required=True)
    a.add_argument("--memories", type=int, required=True)
    a.add_argument("--concurrency", type=int, required=True)
    a.add_argument("--rss-seed-kb", type=int)
    a.add_argument("--rss-recall-kb", type=int)
    a.add_argument("--data-bytes", type=int)
    a.add_argument("--tuning", nargs="*", default=[], help="NAME=value pairs recorded in the result")
    a.add_argument("--out", required=True)
    a.set_defaults(run=assemble)
    t = sub.add_parser("table", help="print the side-by-side markdown table for a results directory")
    t.add_argument("results_dir")
    t.set_defaults(run=table)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
