#!/usr/bin/env python3
"""Latency benchmark for all shodh-memory Python SDK APIs"""

import tempfile
import shutil
import time
from datetime import datetime, timezone
from statistics import mean, stdev

def benchmark(name, func, iterations=10):
    """Run function multiple times and report latency stats"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg = mean(times)
    std = stdev(times) if len(times) > 1 else 0
    min_t = min(times)
    max_t = max(times)

    print(f"{name:40} | {avg:8.2f}ms (std: {std:5.2f}, min: {min_t:6.2f}, max: {max_t:7.2f})")
    return avg

def main():
    test_dir = tempfile.mkdtemp(prefix="shodh_bench_")
    print(f"Benchmark directory: {test_dir}")
    print("=" * 80)

    try:
        from shodh_memory import Memory

        # Create memory system
        mem = Memory(storage_path=f"{test_dir}/db")

        # Seed with some data
        print("\nSeeding 50 memories...")
        for i in range(50):
            mem.remember(
                f"Test memory number {i} about topic {i % 10} with various content",
                memory_type=["Learning", "Decision", "Context", "Error"][i % 4],
                tags=[f"tag{i % 5}", f"group{i % 3}"]
            )
        print("Done seeding.\n")

        print(f"{'API':40} | {'Avg':>8}   (std, min, max)")
        print("-" * 80)

        # === Core APIs ===
        print("\n--- Core APIs ---")

        benchmark("remember(content, type, tags)",
                  lambda: mem.remember("Benchmark memory content", memory_type="Context", tags=["bench"]))

        benchmark("recall(query, limit=10)",
                  lambda: mem.recall("test memory topic", limit=10))

        benchmark("recall(mode='semantic')",
                  lambda: mem.recall("benchmark content", limit=10, mode="semantic"))

        benchmark("recall(mode='associative')",
                  lambda: mem.recall("benchmark content", limit=10, mode="associative"))

        benchmark("recall(mode='hybrid')",
                  lambda: mem.recall("benchmark content", limit=10, mode="hybrid"))

        benchmark("recall_by_tags(tags, limit=20)",
                  lambda: mem.recall_by_tags(tags=["tag1"], limit=20))

        now = datetime.now(timezone.utc)
        start_dt = now.replace(hour=0, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_dt = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        benchmark("recall_by_date(start, end)",
                  lambda: mem.recall_by_date(start=start_dt, end=end_dt, limit=20))

        benchmark("list_memories(limit=50)",
                  lambda: mem.list_memories(limit=50))

        # Get a memory ID for get_memory test
        memories = mem.list_memories(limit=1)
        if memories:
            mem_id = memories[0]['id']
            benchmark("get_memory(id)",
                      lambda: mem.get_memory(mem_id))

        benchmark("get_stats()",
                  lambda: mem.get_stats())

        # === Proactive Context API ===
        print("\n--- Proactive Context API ---")

        benchmark("proactive_context(auto_ingest=True)",
                  lambda: mem.proactive_context("Current conversation about AI memory systems", auto_ingest=True))

        benchmark("proactive_context(auto_ingest=False)",
                  lambda: mem.proactive_context("Query without storing", auto_ingest=False))

        # === Index Health API ===
        print("\n--- Index Health API ---")

        benchmark("verify_index()",
                  lambda: mem.verify_index())

        benchmark("repair_index()",
                  lambda: mem.repair_index())

        benchmark("index_health()",
                  lambda: mem.index_health())

        # === Introspection APIs ===
        print("\n--- Introspection APIs ---")

        benchmark("context_summary(max_items=5)",
                  lambda: mem.context_summary(max_items=5))

        benchmark("brain_state(longterm_limit=50)",
                  lambda: mem.brain_state(longterm_limit=50))

        benchmark("graph_stats()",
                  lambda: mem.graph_stats())

        yesterday = now.replace(hour=0, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
        benchmark("consolidation_report(since)",
                  lambda: mem.consolidation_report(since=yesterday))

        benchmark("consolidation_events(since)",
                  lambda: mem.consolidation_events(since=yesterday))

        # === Forget APIs (single iteration to avoid emptying DB) ===
        print("\n--- Forget APIs (1 iteration each) ---")

        # Add temp memories for forget tests
        temp_id = mem.remember("Temp to delete", memory_type="Context", tags=["delete-me"])

        start = time.perf_counter()
        mem.forget(temp_id)
        print(f"{'forget(id)':40} | {(time.perf_counter() - start)*1000:8.2f}ms")

        mem.remember("Tagged for deletion", memory_type="Context", tags=["bulk-delete"])
        start = time.perf_counter()
        mem.forget_by_tags(tags=["bulk-delete"])
        print(f"{'forget_by_tags(tags)':40} | {(time.perf_counter() - start)*1000:8.2f}ms")

        start = time.perf_counter()
        mem.forget_by_age(days=365)
        print(f"{'forget_by_age(days=365)':40} | {(time.perf_counter() - start)*1000:8.2f}ms")

        start = time.perf_counter()
        mem.forget_by_importance(threshold=0.01)
        print(f"{'forget_by_importance(threshold=0.01)':40} | {(time.perf_counter() - start)*1000:8.2f}ms")

        mem.remember("PATTERN_DELETE_ME", memory_type="Context")
        start = time.perf_counter()
        mem.forget_by_pattern(pattern="PATTERN_DELETE.*")
        print(f"{'forget_by_pattern(pattern)':40} | {(time.perf_counter() - start)*1000:8.2f}ms")

        start = time.perf_counter()
        mem.forget_by_date(start="2024-01-01T00:00:00Z", end="2024-12-31T23:59:59Z")
        print(f"{'forget_by_date(start, end)':40} | {(time.perf_counter() - start)*1000:8.2f}ms")

        # === Storage ===
        print("\n--- Storage ---")

        benchmark("flush()",
                  lambda: mem.flush())

        print("\n" + "=" * 80)
        print("Benchmark complete!")

    finally:
        shutil.rmtree(test_dir)
        print(f"Cleaned up: {test_dir}")

if __name__ == "__main__":
    main()
