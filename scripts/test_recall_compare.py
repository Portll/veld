"""Tests for the paired, conversation-clustered run comparison.

`recall_compare` is the evidence a default change rests on, so the arithmetic is the claim: the
pairing must refuse runs over different cases, the clustering must group a conversation's cases,
and an interval must widen when cases inside a conversation move together. Runs standalone so it
needs nothing installed, and is still collectable by pytest if that is present:

    python scripts/test_recall_compare.py
    python -m pytest scripts/test_recall_compare.py
"""

import json
import random
import tempfile
from pathlib import Path

from recall_compare import (
    PairingError,
    bootstrap_case,
    bootstrap_cluster,
    compare,
    conversation_of,
    load_cases,
    main,
    pair,
    quantile,
    randomisation_case,
    randomisation_cluster,
)


def case(case_id, p1, category="single_hop"):
    return {
        "case_id": case_id,
        "category": category,
        "p_at_1": p1,
        "mrr": p1,
        "recall_at_k": p1,
        "ndcg_at_k": p1,
    }


def write(path, cases, layer="full"):
    path.write_text(json.dumps({layer: cases}), encoding="utf-8")


class TestConversationOf:
    def test_locomo_ids_group_by_conversation(self):
        assert conversation_of("conv-42_q0") == "conv-42"
        assert conversation_of("conv-42_q117") == "conv-42"
        assert conversation_of("conv-30_q5") == "conv-30"

    def test_an_id_without_a_question_suffix_is_its_own_cluster(self):
        # A non-LoCoMo suite must degrade to per-case, not lump every case together.
        assert conversation_of("smoke-001") == "smoke-001"
        assert conversation_of("conv-42_qx") == "conv-42_qx"
        assert conversation_of("_q3") == "_q3"


class TestPairing:
    def test_refuses_different_case_sets(self):
        try:
            pair({"a_q1": case("a_q1", 1)}, {"a_q2": case("a_q2", 1)})
        except PairingError as e:
            assert "only in the baseline" in str(e)
        else:
            raise AssertionError("pairing two different case sets must refuse")

    def test_pairs_by_id_not_by_position(self):
        base = {"a_q1": case("a_q1", 0), "a_q2": case("a_q2", 1)}
        cand = {"a_q2": case("a_q2", 1), "a_q1": case("a_q1", 1)}
        pairs = pair(base, cand)
        assert [p[0] for p in pairs] == ["a_q1", "a_q2"]
        assert pairs[0][1]["p_at_1"] == 0 and pairs[0][2]["p_at_1"] == 1

    def test_load_refuses_a_duplicated_case(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.json"
            write(path, [case("a_q1", 1), case("a_q1", 0)])
            try:
                load_cases(path, "full")
            except PairingError as e:
                assert "twice" in str(e)
            else:
                raise AssertionError("a duplicated case id must refuse")

    def test_load_refuses_a_missing_layer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.json"
            write(path, [case("a_q1", 1)], layer="bm25")
            try:
                load_cases(path, "full")
            except PairingError as e:
                assert "no layer" in str(e)
            else:
                raise AssertionError("a missing layer must refuse")


class TestQuantile:
    def test_interpolates(self):
        assert quantile([0.0, 1.0], 0.5) == 0.5
        assert quantile([0.0, 1.0, 2.0], 0.0) == 0.0
        assert quantile([0.0, 1.0, 2.0], 1.0) == 2.0


class TestRandomisation:
    def test_identical_runs_have_p_one(self):
        rng = random.Random(1)
        assert randomisation_case([0.0] * 50, 1000, rng) == 1.0
        assert randomisation_cluster([0.0] * 5, 1000, rng)[0] == 1.0

    def test_a_uniform_gain_over_many_cases_is_significant_per_case(self):
        p = randomisation_case([1.0] * 40, 4000, random.Random(2))
        assert p < 0.001, p

    def test_exact_cluster_test_bottoms_out_at_two_over_two_to_the_k(self):
        # Nine conversations that all moved the same way: the most extreme of 512 patterns, and
        # its mirror image, so p = 2/512. No amount of per-case agreement can go below that.
        p, exact = randomisation_cluster([3.0] * 9, 1000, random.Random(3))
        assert exact
        assert abs(p - 2 / 512) < 1e-12, p

    def test_cluster_test_samples_beyond_the_exact_limit(self):
        p, exact = randomisation_cluster([1.0] * 20, 2000, random.Random(4))
        assert not exact
        assert p < 0.01


class TestClusteringWidensTheInterval:
    def test_correlated_conversations_widen_the_ci(self):
        # Ten conversations of 30 cases. Within a conversation every case moves the same way;
        # across conversations the direction varies. Per case the gain looks tight; per
        # conversation it must not.
        rng = random.Random(5)
        diffs, clusters = [], []
        for k in range(10):
            d = 1.0 if k % 3 else -1.0
            diffs.extend([d] * 30)
            clusters.append((d * 30, 30))
        lo_c, hi_c = bootstrap_case(diffs, 3000, rng, 0.05)
        lo_k, hi_k = bootstrap_cluster(clusters, 3000, rng, 0.05)
        assert (hi_k - lo_k) > 3 * (hi_c - lo_c), ((lo_c, hi_c), (lo_k, hi_k))
        assert lo_c > 0, "per case the gain looks real"
        assert lo_k < 0, "per conversation it cannot be told from zero"

    def test_cluster_mean_weights_by_size(self):
        # One conversation of 90 cases at +1 and one of 10 at -1: the suite mean is +0.8, and
        # every resample is a ratio of sums, so no resample can exceed +1 or fall below -1, and
        # the interval contains the full-suite mean.
        lo, hi = bootstrap_cluster([(90.0, 90), (-10.0, 10)], 4000, random.Random(6), 0.05)
        assert -1.0 <= lo <= 0.8 <= hi <= 1.0, (lo, hi)


class TestCompare:
    def test_counts_and_means(self):
        pairs = [
            ("a_q1", case("a_q1", 0), case("a_q1", 1)),
            ("a_q2", case("a_q2", 1), case("a_q2", 0)),
            ("b_q1", case("b_q1", 0), case("b_q1", 1)),
            ("b_q2", case("b_q2", 1), case("b_q2", 1)),
        ]
        r = compare(pairs, "p_at_1", 500, "s", 0.05)
        assert r.mean_baseline == 0.5
        assert r.mean_candidate == 0.75
        assert r.delta == 0.25
        assert (r.better, r.worse) == (2, 1)
        assert r.exact_conversation
        assert r.p_conversation == 1.0  # two conversations, +1 and +1 - 1 + 1: flips can match it

    def test_one_conversation_reports_no_cluster_interval(self):
        pairs = [(f"a_q{k}", case(f"a_q{k}", 0), case(f"a_q{k}", 1)) for k in range(5)]
        r = compare(pairs, "p_at_1", 200, "s", 0.05)
        assert r.ci_conversation is None and r.p_conversation is None

    def test_is_deterministic_for_a_seed(self):
        pairs = [(f"c{k % 4}_q{k}", case(f"c{k % 4}_q{k}", k % 2), case(f"c{k % 4}_q{k}", (k // 2) % 2)) for k in range(40)]
        assert compare(pairs, "mrr", 800, "7", 0.05) == compare(pairs, "mrr", 800, "7", 0.05)


class TestMain:
    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = Path(d) / "a.json", Path(d) / "b.json", Path(d) / "c.json"
            write(a, [case("x_q1", 0), case("x_q2", 1)])
            write(b, [case("x_q1", 1), case("x_q2", 1)])
            write(c, [case("y_q1", 1)])
            assert main([str(a), str(b), "--resamples", "200"]) == 0
            assert main([str(a), str(c), "--resamples", "200"]) == 2


if __name__ == "__main__":
    # A plain runner, because this repo ships no python test dependency and a
    # test that cannot be run is documentation.
    failed = 0
    for cls in (
        TestConversationOf,
        TestPairing,
        TestQuantile,
        TestRandomisation,
        TestClusteringWidensTheInterval,
        TestCompare,
        TestMain,
    ):
        instance = cls()
        for name in sorted(n for n in dir(cls) if n.startswith("test_")):
            try:
                getattr(instance, name)()
                print(f"  ok   {cls.__name__}.{name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL {cls.__name__}.{name}: {e}")
    print()
    print(f"{'FAILED' if failed else 'all passed'} ({failed} failure(s))")
    raise SystemExit(1 if failed else 0)
