"""Tests for the check that both storage engines score the same.

Runs standalone, like test_recall_diff.py:

    python scripts/test_recall_engines_equal.py
"""

from recall_engines_equal import verdict

AVX2 = {"class": "x86_64-avx2-fma", "cpu_model": "AMD EPYC 7763 64-Core Processor"}
AVX512 = {"class": "x86_64-avx512f+avx512bw+avx512dq+avx512vl+avx512vnni", "cpu_model": "Xeon"}


def report(p_at_1=0.35, kernel=AVX2, tape=None):
    r = {
        "embedder": "minilm-l6-v2",
        "layers": {"full": {"p@1": p_at_1, "recall@10": 0.5, "latency_p50_ms": 900.0}},
        "by_category": {"temporal": {"p@1": 0.5, "case_count": 30}},
    }
    if kernel is not None:
        r["kernel"] = kernel
    if tape is not None:
        r["model_tape"] = tape
    return r


class TestEnginesEqual:
    def test_equal_scores_pass_and_name_both_classes(self):
        code, lines = verdict(report(), report(kernel=AVX512), "rocksdb", "redb")
        assert code == 0
        assert lines[0] == f"CPU kernel class: rocksdb {AVX2['class']}, redb {AVX512['class']}"

    def test_latency_is_not_compared(self):
        other = report()
        other["layers"]["full"]["latency_p50_ms"] = 400.0
        assert verdict(report(), other)[0] == 0

    def test_a_difference_on_one_class_fails(self):
        code, lines = verdict(report(), report(p_at_1=0.34))
        assert code == 1
        assert any(l.startswith("layers.full.p@1:") for l in lines)

    def test_a_difference_across_classes_is_not_attributed_to_the_engine(self):
        for right in (report(p_at_1=0.34, kernel=AVX512), report(p_at_1=0.34, kernel=None)):
            code, lines = verdict(report(), right)
            assert code == 2
            assert "cannot be attributed to the storage engine" in lines[-1]

    def test_one_tape_makes_the_class_irrelevant(self):
        tape = "a" * 64
        code, _ = verdict(report(tape=tape), report(p_at_1=0.34, kernel=AVX512, tape=tape))
        assert code == 1

    def test_different_model_outputs_are_refused(self):
        code, lines = verdict(report(tape="a" * 64), report())
        assert code == 2
        assert lines[0].startswith("model outputs differ")


if __name__ == "__main__":
    failed = 0
    for name in sorted(n for n in dir(TestEnginesEqual) if n.startswith("test_")):
        try:
            getattr(TestEnginesEqual(), name)()
            print(f"  ok   {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if failed else 'all passed'} ({failed} failure(s))")
    raise SystemExit(1 if failed else 0)
