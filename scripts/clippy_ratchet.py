#!/usr/bin/env python3
"""Clippy warning ratchet: fail when a feature set's warning count rises above its baseline.

  clippy_ratchet.py <feature-set> <clippy-output.json>

The input is `cargo clippy --message-format=json` output. Every diagnostic is printed in its
rendered form, so the log reads as clippy's own would. A warning is counted once per distinct
rendered text: cargo reports a lib warning again for the lib's test target, and a module that
several test crates share once per crate, and those are one place to fix.

The counts in clippy_baseline.json were measured with the toolchain it names. A new Rust
release adds lints, so a run on any other toolchain is refused rather than compared: bump the
toolchain in ci.yml and re-measure the baseline in the same change.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASELINE = Path(__file__).with_name("clippy_baseline.json")
SUMMARY = re.compile(r"^\d+ warnings? emitted")


def rustc_version():
    out = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True).stdout
    return out.split()[1]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    feature_set, output = sys.argv[1], Path(sys.argv[2])
    baseline = json.loads(BASELINE.read_text())
    if feature_set not in baseline["counts"]:
        sys.exit(f"{feature_set}: no baseline in {BASELINE.name}; known: {sorted(baseline['counts'])}")

    toolchain = rustc_version()
    if toolchain != baseline["toolchain"]:
        print(f"::error::{feature_set}: clippy ran on rustc {toolchain}, and {BASELINE.name} was "
              f"measured on {baseline['toolchain']}. Lint sets differ between releases, so the "
              f"counts do not compare: re-measure the baseline for the new toolchain.")
        return 1

    warnings = set()
    errors = 0
    for line in output.read_text().splitlines():
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if message.get("reason") != "compiler-message":
            continue
        diagnostic = message["message"]
        if SUMMARY.match(diagnostic["message"]):
            continue
        rendered = diagnostic.get("rendered") or diagnostic["message"]
        if diagnostic["level"] == "warning":
            if rendered not in warnings:
                warnings.add(rendered)
                print(rendered, end="" if rendered.endswith("\n") else "\n")
        elif diagnostic["level"] == "error":
            errors += 1
            print(rendered, end="" if rendered.endswith("\n") else "\n")

    count, allowed = len(warnings), baseline["counts"][feature_set]
    status = "passed"
    code = 0
    if count > allowed:
        status = "warnings rose"
        code = 1
        print(f"::error::{feature_set}: {count} clippy warnings, baseline {allowed}. Fix the new "
              f"warnings; the count may only fall.")
    elif count < allowed:
        print(f"::notice::{feature_set}: {count} clippy warnings, baseline {allowed}. Lower "
              f"{BASELINE.name} to {count} so the slack cannot be spent again.")
    if errors:
        status = "errors"
        code = 1
    print(f"{feature_set}: {count} warnings (baseline {allowed}), {errors} errors")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"| {feature_set} | {count} | {allowed} | {errors} | {status} |\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
