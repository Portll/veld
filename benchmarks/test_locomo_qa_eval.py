"""
Offline tests for locomo_qa_eval.py and locomo_qa_mem0.py: no network, no server, no mem0 or
tiktoken import (token counts are stubbed). From the repository root:

    ~/.venvs/veld-bench/bin/python -m unittest discover -s benchmarks -p 'test_locomo_qa*.py'
"""

import argparse
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import locomo_qa_eval as qa
import locomo_qa_mem0 as m0

WORDS = lambda text: len(text.split())  # noqa: E731 - a stand-in token count


def item(content, created_at="2023-05-08T13:56:00Z", dia_ids=(), speaker="Caroline"):
    return {"id": content[:8], "dia_ids": list(dia_ids), "speaker": speaker,
            "created_at": created_at, "score": 0.5, "content": content}


class EvidenceTests(unittest.TestCase):
    def test_packed_and_padded_ids_name_their_turns(self):
        self.assertEqual(qa.evidence_ids(["D8:6; D9:17"]), ["D8:6", "D9:17"])
        self.assertEqual(qa.evidence_ids(["D9:1 D4:4 D4:6"]), ["D9:1", "D4:4", "D4:6"])
        self.assertEqual(qa.evidence_ids(["D30:05"]), ["D30:5"])
        self.assertEqual(qa.evidence_ids(["D1:3", "D1:3", 7]), ["D1:3"])
        self.assertEqual(qa.evidence_ids([]), [])

    def test_retrieval_set_needs_an_exact_turn_id(self):
        turns = {"D1:3", "D8:6", "D30:5"}
        self.assertTrue(qa.in_retrieval_set(["D1:3"], turns))
        self.assertFalse(qa.in_retrieval_set([], turns))
        self.assertFalse(qa.in_retrieval_set(["D8:6; D9:17"], turns))
        self.assertFalse(qa.in_retrieval_set(["D30:05"], turns))

    def test_conversation_questions_mark_the_retrieval_set(self):
        conv = {"sample_id": "conv-x", "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"},
                          {"speaker": "B", "dia_id": "D1:2", "text": "yo"}],
            "session_1_date_time": "1:56 pm on 8 May, 2023"},
            "qa": [{"question": "q0", "answer": "a", "category": 4, "evidence": ["D1:1"]},
                   {"question": "q1", "answer": 2023, "category": 2, "evidence": ["D1:02"]},
                   {"question": "q2", "answer": "a", "category": 5, "evidence": ["D1:1"]},
                   {"question": "q3", "answer": "a", "category": 3, "evidence": []}]}
        rows = qa.conversation_questions(conv)
        self.assertEqual([r["qid"] for r in rows], ["conv-x_q0", "conv-x_q1", "conv-x_q3"])
        self.assertEqual([r["in_retrieval_set"] for r in rows], [True, False, False])
        self.assertEqual(rows[1]["evidence"], ["D1:2"])
        self.assertEqual(rows[1]["gold"], "2023")


class BudgetTests(unittest.TestCase):
    def test_no_budget_keeps_every_line(self):
        items = [item("one two"), item("three four five")]
        self.assertEqual(qa.fit_context(items, None, WORDS), [3, 4])

    def test_budget_keeps_the_ranked_prefix(self):
        items = [item("a b c"), item("d e f g h i j"), item("k")]
        # "[2023-05-08] a b c" is 4 words; the second line (8) would pass 10, and the third,
        # which would fit, is not taken out of rank order.
        self.assertEqual(qa.fit_context(items, 10, WORDS), [4])
        self.assertEqual(qa.fit_context(items, 3, WORDS), [])
        self.assertEqual(qa.fit_context(items, 14, WORDS), [4, 8, 2])

    def test_settle_context_counts_and_measures_evidence_on_what_is_shown(self):
        row = {"qid": "q", "evidence": ["D2:1"],
               "retrieved": [item("a b c", dia_ids=["D1:1"]), item("d e", dia_ids=["D2:1"])]}
        qa.settle_context(row, 4, WORDS, maps_turns=True)
        self.assertEqual((row["context_n"], row["context_tokens"], row["retrieved_tokens"]),
                         (1, 4, 7))
        self.assertFalse(row["evidence_hit"])
        qa.settle_context(row, None, WORDS, maps_turns=True)
        self.assertTrue(row["evidence_hit"])
        qa.settle_context(row, None, WORDS, maps_turns=False)
        self.assertIsNone(row["evidence_hit"])

    def test_reader_prompt_shows_only_the_kept_lines(self):
        row = {"qid": "q", "question": "Where?", "context_n": 1,
               "retrieved": [item("Caroline: kept"), item("Melanie: dropped", speaker="Melanie")]}
        prompt = qa.reader_prompt(row, ("Caroline", "Melanie"))
        self.assertIn("[2023-05-08] Caroline: kept", prompt)
        self.assertNotIn("dropped", prompt)


class WilsonTests(unittest.TestCase):
    def test_known_values(self):
        lo, hi = qa.wilson(0.5, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        lo, hi = qa.wilson(0.0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = qa.wilson(1.0, 10)
        self.assertAlmostEqual(lo, 0.7225, places=3)
        self.assertAlmostEqual(hi, 1.0, places=9)

    def test_no_questions_no_interval(self):
        self.assertIsNone(qa.wilson(0.0, 0))


def run(label=None, calls=()):
    return {"run": 0, "answer": "x" if label else None, "label": label, "calls": list(calls)}


def error_call(error, fatal=False, kind="reader"):
    return {"kind": kind, "error": error, "fatal": fatal, "stop_reason": None, "fallback": False,
            "request_id": None}


class StateTests(unittest.TestCase):
    def test_states(self):
        self.assertEqual(qa.unit_state(run("CORRECT")), "judged")
        self.assertEqual(qa.unit_state(run()), "not_attempted")
        self.assertEqual(qa.unit_state(run(calls=[error_call("HTTP 401: no", fatal=True)])),
                         "fatal")
        self.assertEqual(qa.unit_state(run(calls=[error_call("length: output truncated")])),
                         "persistent")
        self.assertEqual(qa.unit_state(run(calls=[error_call("refusal (content_filter)")])),
                         "persistent")
        self.assertEqual(qa.unit_state(run(calls=[error_call("connection error after retries")])),
                         "retryable")
        self.assertEqual(qa.unit_state(run(calls=[error_call("judge output is not the label "
                                                             "schema", kind="judge")])),
                         "retryable")

    def test_quota_waits(self):
        body = ("Rate limit reached for model `m` on tokens per day (TPD): Limit 200000, Used "
                "199047, Requested 1413. Please try again in 3m18.72s.")
        self.assertAlmostEqual(qa.quota_wait_s(body), 198.72)
        self.assertEqual(qa.quota_wait_s("try again in 1h2m3s"), 3723)
        self.assertIsNone(qa.quota_wait_s("no wait named"))

    def test_a_daily_quota_with_a_short_wait_is_not_terminal(self):
        def error(message, headers=None):
            return SimpleNamespace(message=message,
                                   response=SimpleNamespace(headers=headers or {}))
        tpd = "Limit 200000 on tokens per day (TPD). Please try again in 3m18.72s."
        self.assertIsNone(qa.rate_limit_terminal(error(tpd)))
        self.assertIn("per-day", qa.rate_limit_terminal(error("tokens per day (TPD) spent")))
        self.assertIn("wait cap", qa.rate_limit_terminal(error("x", {"retry-after": "5000"})))


def results_with(rows):
    return {"rows": {r["qid"]: r for r in rows},
            "config": {"price_per_mtok": {}}}


def row(qid, category, labels, in_set=True, context=None):
    return {"qid": qid, "category": category, "in_retrieval_set": in_set, "evidence_hit": None,
            "recall_ms": 1.0, "context_tokens": context, "context_n": 1 if context else None,
            "retrieved_tokens": context,
            "runs": [run(lbl) if lbl else run(calls=[error_call("length: x")]) for lbl in labels]}


class AggregateTests(unittest.TestCase):
    def test_interval_reasons_subset_and_status(self):
        rows = [row("a", "single_hop", ["CORRECT"], context=100),
                row("b", "single_hop", ["WRONG"], context=300),
                row("c", "temporal", ["CORRECT"], in_set=False, context=200),
                row("d", "temporal", [None], context=50)]
        results = results_with(rows)
        agg = qa.summarise(results, ["a", "b", "c", "d"], 1)
        self.assertEqual(results["status"], "incomplete")
        self.assertEqual(results["unjudged"], 1)
        self.assertEqual(agg["unjudged_reasons"], {"persistent": 1})
        overall = agg["j"]["overall"]
        self.assertAlmostEqual(overall["mean"], 2 / 3)
        lo, hi = overall["ci95"]
        self.assertLess(lo, 2 / 3)
        self.assertGreater(hi, 2 / 3)
        self.assertEqual(agg["retrieval_set"]["questions"], 3)
        self.assertEqual(agg["retrieval_set"]["excluded"], ["c"])
        self.assertAlmostEqual(agg["retrieval_set"]["j"]["mean"], 0.5)
        self.assertEqual(agg["context_tokens"]["mean"], 162.5)
        self.assertEqual(agg["context_tokens"]["max"], 300)

    def test_complete_when_everything_is_judged(self):
        results = results_with([row("a", "single_hop", ["CORRECT", "WRONG"])])
        agg = qa.summarise(results, ["a"], 2)
        self.assertEqual(results["status"], "complete")
        self.assertEqual(agg["j"]["overall"]["mean"], 0.5)
        self.assertEqual(agg["j"]["overall"]["ci95"], qa.wilson(0.5, 1))

    def test_runs_never_started_count_as_not_attempted(self):
        results = results_with([row("a", "single_hop", ["CORRECT"])])
        agg = qa.summarise(results, ["a"], 3)
        self.assertEqual(results["status"], "incomplete")
        self.assertEqual(agg["unjudged_reasons"], {"not_attempted": 2})


class RetryTests(unittest.TestCase):
    """api_phase sends a retryable failure again and leaves a persistent one alone."""

    def test_retryable_errors_are_sent_again(self):
        sends = {"q1": 0, "q2": 0}

        def fake_run_unit(client, llm, row, run, prompt, abort):
            sends[row["qid"]] += 1
            if row["qid"] == "q1" and sends["q1"] < 3:
                run["calls"].append(error_call("connection error after retries"))
                return run, False
            if row["qid"] == "q2":
                run["calls"].append(error_call("length: output truncated at 10 tokens"))
                return run, False
            run["answer"], run["label"] = "a", "CORRECT"
            return run, False

        rows = [{"qid": q, "conversation": "c", "category": "single_hop", "question": "?",
                 "gold": "g", "retrieved": [], "runs": []} for q in ("q1", "q2")]
        results = results_with(rows)
        args = argparse.Namespace(runs=1, concurrency=2, attempts=3)
        original = qa.run_unit
        qa.run_unit = fake_run_unit
        try:
            with tempfile.TemporaryDirectory() as d:
                qa.api_phase(args, results, ["q1", "q2"], {"c": ("A", "B")},
                             Path(d) / "r.json", None, {"reader": {"model": "m"},
                                                         "judge": {"model": "m"}},
                             pause=lambda s: None)
        finally:
            qa.run_unit = original
        self.assertEqual(sends, {"q1": 3, "q2": 1})
        self.assertEqual(results["rows"]["q1"]["runs"][0]["label"], "CORRECT")
        self.assertEqual(qa.unit_state(results["rows"]["q2"]["runs"][0]), "persistent")
        self.assertEqual(results["api_phase"][-1]["rounds"], 3)


class Mem0PureTests(unittest.TestCase):
    def test_session_chunks_carry_the_session_time(self):
        turns = [{"speaker": "A", "dia_id": f"D1:{i}", "text": f"t{i}"} for i in range(1, 6)]
        start = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
        content = lambda t: f"{t['speaker']}: {t['text']}"  # noqa: E731
        whole = m0.session_chunks(turns, start, 0, content)
        self.assertEqual(len(whole), 1)
        self.assertEqual(whole[0]["messages"][0],
                         {"role": "system",
                          "content": "These messages were exchanged at 1:56 PM on 8 May 2023."})
        self.assertEqual(whole[0]["messages"][1:],
                         [{"role": "user", "content": f"A: t{i}"} for i in range(1, 6)])
        self.assertEqual(whole[0]["created_at"], "2023-05-08T13:56:00Z")
        self.assertIn("2023-05-08", whole[0]["instructions"])
        pairs = m0.session_chunks(turns, start, 2, content)
        self.assertEqual([c["dia_ids"] for c in pairs], [["D1:1", "D1:2"], ["D1:3", "D1:4"],
                                                          ["D1:5"]])
        self.assertEqual(pairs[2]["created_at"], "2023-05-08T13:56:04Z")

    def test_attribution_is_the_first_named_speaker(self):
        speakers = ("Caroline", "Melanie")
        self.assertEqual(m0.attribute_speaker("Melanie painted with Caroline", speakers),
                         ("Melanie", True))
        self.assertEqual(m0.attribute_speaker("caroline went hiking", speakers),
                         ("Caroline", True))
        self.assertEqual(m0.attribute_speaker("Carolines are rare", speakers),
                         ("Caroline", False))
        self.assertEqual(m0.attribute_speaker("Went to a support group", speakers),
                         ("Caroline", False))


class StubMemory:
    def __init__(self, version):
        self.version = version
        self.adds = []
        self.searches = []
        self.failures = 0

    def add(self, messages, **kwargs):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("host went away")
        self.adds.append((messages, kwargs))
        return {"results": [{"id": "m1", "memory": "fact", "event": "ADD"}]}

    def search(self, query, **kwargs):
        self.searches.append((query, kwargs))
        return {"results": [
            {"id": "m1", "memory": "Melanie painted a sunrise", "score": 0.9,
             "created_at": "2023-05-08T13:56:00+00:00"},
            {"id": "m2", "memory": "Likes hiking", "score": 0.4,
             "created_at": "2023-06-01T10:00:00+00:00"}]}

    def get_all(self, **kwargs):
        self.get_all_kwargs = kwargs
        return {"results": [{"id": "m1"}, {"id": "m2"}]}


def stub_system(version, memory, sleeps=None):
    return m0.Mem0System(provider="lmstudio", base_url="http://127.0.0.1:11234/v1",
                         api_key="k", llm_model="qwen", llm_effort="none", llm_max_tokens=100,
                         embedder_model="nomic", embedding_dims=768, batch_turns=0,
                         memory=memory, version=version,
                         sleep=(sleeps.append if sleeps is not None else lambda s: None))


SESSIONS = [(1, [{"speaker": "Caroline", "dia_id": "D1:1", "text": "hi"},
                 {"speaker": "Melanie", "dia_id": "D1:2", "text": "hey"}],
             datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)),
            (2, [{"speaker": "Melanie", "dia_id": "D2:1", "text": "back"}],
             datetime(2023, 5, 25, 9, 0, tzinfo=timezone.utc))]


class Mem0AdapterTests(unittest.TestCase):
    def test_v2_ingest_and_search(self):
        memory = StubMemory("2.2.1")
        system = stub_system("2.2.1", memory)
        lines = []
        stats = system.ingest("locomo_conv-x", SESSIONS, qa.turn_content, lines.append)
        self.assertEqual((stats["sessions"], stats["turns"], stats["adds"]), (2, 3, 2))
        self.assertEqual(len(lines), 2)
        self.assertIn("session 2 (2 of 2) added, 2 memory events so far", lines[1])
        self.assertEqual(stats["events"], {"ADD": 2})
        self.assertEqual(stats["stored_memories"], 2)
        messages, kwargs = memory.adds[0]
        self.assertEqual(kwargs["user_id"], "locomo_conv-x")
        self.assertEqual(kwargs["metadata"], {"created_at": "2023-05-08T13:56:00Z",
                                              "locomo_session": 1})
        self.assertIn("2023-05-08", kwargs["prompt"])
        self.assertEqual([m["content"] for m in messages[1:]], ["Caroline: hi", "Melanie: hey"])
        self.assertEqual(memory.get_all_kwargs["filters"], {"user_id": "locomo_conv-x"})
        found = system.search("locomo_conv-x", "What did Melanie paint?", 1,
                              ("Caroline", "Melanie"))
        self.assertEqual(memory.searches[0][1], {"filters": {"user_id": "locomo_conv-x"},
                                                 "top_k": 1})
        self.assertEqual(len(found["retrieved"]), 1)
        first = found["retrieved"][0]
        self.assertEqual((first["speaker"], first["dia_ids"], first["content"]),
                         ("Melanie", [], "Melanie painted a sunrise"))
        self.assertIsNone(found["evidence_hit"])
        self.assertEqual(found["unattributed"], 0)

    def test_v1_call_shapes(self):
        memory = StubMemory("1.0.11")
        system = stub_system("1.0.11", memory)
        system.ingest("u", SESSIONS[:1], qa.turn_content)
        self.assertNotIn("prompt", memory.adds[0][1])
        self.assertEqual(memory.get_all_kwargs, {"user_id": "u", "limit": 100_000})
        found = system.search("u", "q", 5, ("Caroline", "Melanie"))
        self.assertEqual(memory.searches[0][1], {"user_id": "u", "limit": 5})
        self.assertEqual(found["unattributed"], 1)
        self.assertEqual(found["retrieved"][1]["speaker"], "Caroline")

    def test_add_retries_then_names_the_session(self):
        memory = StubMemory("2.2.1")
        memory.failures = 2
        sleeps = []
        system = stub_system("2.2.1", memory, sleeps)
        system.add({"messages": [], "created_at": "t", "instructions": "i"}, "u", 3)
        self.assertEqual(sleeps, [15, 30])
        memory.failures = m0.ADD_ATTEMPTS
        with self.assertRaisesRegex(RuntimeError, "u session 4: mem0 add failed 4 times"):
            system.add({"messages": [], "created_at": "t", "instructions": "i"}, "u", 4)

    def test_config_uses_the_lmstudio_provider(self):
        config = stub_system("2.2.1", StubMemory("2.2.1")).config("/tmp/x")
        self.assertEqual(config["llm"]["provider"], "lmstudio")
        self.assertNotIn("lmstudio_response_format", config["llm"]["config"])
        self.assertEqual(config["llm"]["config"]["temperature"], 0)
        self.assertEqual(config["embedder"]["config"]["lmstudio_base_url"],
                         "http://127.0.0.1:11234/v1")
        self.assertEqual(config["vector_store"]["config"]["embedding_model_dims"], 768)


class FormatTests(unittest.TestCase):
    def test_each_call_gets_the_schema_mem0_parses(self):
        system_first = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        user_only = [{"role": "user", "content": "u"}]
        name = lambda v, m: m0.lmstudio_format(v, m)["json_schema"]["name"]  # noqa: E731
        self.assertEqual(name("2.2.1", system_first), "mem0_extraction")
        self.assertEqual(name("1.0.11", system_first), "mem0_facts")
        self.assertEqual(name("1.0.11", user_only), "mem0_actions")
        fmt = m0.lmstudio_format("2.2.1", system_first)
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertEqual(fmt["json_schema"]["schema"]["required"], ["memory"])


class AccountingTests(unittest.TestCase):
    def test_counts_replies_and_injects_reasoning_effort(self):
        seen = []

        def reply(content, finish="stop"):
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason=finish,
                                         message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

        replies = [reply('{"memory": []}'), reply("not json"), reply("", "length")]
        formats = []
        inner = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **p: seen.append(p) or replies.pop(0))),
            embeddings=SimpleNamespace(create=lambda **p: "vectors"),
            base_url="u")
        log = m0.CallLog()
        client = m0.AccountingClient(inner, log, {"reasoning_effort": "none"},
                                     lambda messages: formats.append(messages) or {"type": "x"})
        for _ in range(3):
            client.chat.completions.create(model="m", messages=[],
                                           response_format={"type": "json_object"})
        self.assertEqual(client.embeddings.create(input=["a", "b"]), "vectors")
        self.assertEqual(client.base_url, "u")
        self.assertEqual(seen[0]["extra_body"], {"reasoning_effort": "none"})
        self.assertEqual(seen[0]["response_format"], {"type": "x"})
        self.assertEqual(len(formats), 3)
        snap = log.snapshot()
        self.assertEqual({k: snap["llm"][k] for k in ("calls", "unparseable", "truncated",
                                                       "empty", "nothing_extracted",
                                                       "input_tokens")},
                         {"calls": 3, "unparseable": 1, "truncated": 1, "empty": 1,
                          "nothing_extracted": 1, "input_tokens": 30})
        self.assertEqual((snap["embedding"]["calls"], snap["embedding"]["inputs"]), (1, 2))

    def test_errors_are_counted_and_raised(self):
        def boom(**params):
            raise TimeoutError("slow host")

        inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)),
                                embeddings=SimpleNamespace(create=boom))
        log = m0.CallLog()
        client = m0.AccountingClient(inner, log, {})
        with self.assertRaises(TimeoutError):
            client.chat.completions.create(model="m", messages=[])
        self.assertEqual(log.snapshot()["llm"]["errors"], 1)
        self.assertIn("slow host", log.snapshot()["llm"]["last_errors"][0])


class Mem0PhaseTests(unittest.TestCase):
    def test_rows_settle_without_evidence_and_save_per_conversation(self):
        memory = StubMemory("2.2.1")
        system = stub_system("2.2.1", memory)
        conv = {"sample_id": "conv-x", "conversation": {
            "speaker_a": "Caroline", "speaker_b": "Melanie",
            "session_1": SESSIONS[0][1], "session_1_date_time": "1:56 pm on 8 May, 2023"}}
        todo = [{"qid": "conv-x_q0", "conversation": "conv-x", "question": "paint?",
                 "evidence": ["D1:1"], "category": "single_hop"}]
        results = {"rows": {}, "ingest": {}, "config": {}, "system": {"name": "mem0"}}
        args = argparse.Namespace(top_k=2, context_budget=None)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            qa.mem0_retrieval_phase(args, {"conv-x": conv}, results, todo, out, {}, WORDS, None,
                                    system=system)
            self.assertTrue(out.exists())
        r = results["rows"]["conv-x_q0"]
        self.assertIsNone(r["evidence_hit"])
        self.assertEqual(r["context_n"], 2)
        self.assertEqual(results["system"]["version"], "2.2.1")
        self.assertEqual(results["ingest"]["conv-x"]["unattributed_memories"], 1)


if __name__ == "__main__":
    unittest.main()
