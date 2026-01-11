#!/usr/bin/env python3
"""Comprehensive test of all shodh-memory Python SDK APIs"""

import tempfile
import shutil
import sys
from datetime import datetime, timezone

def test_all_apis():
    # Create temp directory for test
    test_dir = tempfile.mkdtemp(prefix="shodh_test_")
    print(f"Test directory: {test_dir}")

    passed = 0
    failed = 0

    def test(name, func):
        nonlocal passed, failed
        try:
            result = func()
            print(f"[PASS] {name}")
            if result:
                print(f"       -> {result}")
            passed += 1
            return True
        except Exception as e:
            print(f"[FAIL] {name}")
            print(f"       ERROR: {e}")
            failed += 1
            return False

    try:
        # Import test
        test("Import shodh_memory", lambda: __import__('shodh_memory'))
        from shodh_memory import Memory, __version__

        test("Check version", lambda: f"v{__version__}")

        # Create memory system
        mem = None
        def create_memory():
            nonlocal mem
            mem = Memory(storage_path=f"{test_dir}/db")
            return f"Created at {test_dir}/db"
        test("Create Memory instance", create_memory)

        # === Core APIs ===
        print("\n--- Core APIs ---")

        # remember()
        mem_id = None
        def remember_test():
            nonlocal mem_id
            mem_id = mem.remember("Test memory for API validation", memory_type="Learning", tags=["test", "api"])
            return f"ID: {mem_id[:8]}..."
        test("remember(content, memory_type, tags)", remember_test)

        # remember() with metadata
        test("remember() with metadata",
             lambda: mem.remember("Memory with metadata", memory_type="Context", metadata={"key": "value"}))

        # recall()
        def recall_test():
            results = mem.recall("test memory", limit=5)
            return f"Found {len(results)} memories"
        test("recall(query, limit)", recall_test)

        # recall() with mode
        def recall_mode_test():
            results = mem.recall("test", limit=5, mode="semantic")
            return f"Semantic: {len(results)} results"
        test("recall(mode='semantic')", recall_mode_test)

        # recall_by_tags()
        def recall_tags_test():
            results = mem.recall_by_tags(tags=["test"], limit=10)
            return f"Found {len(results)} by tags"
        test("recall_by_tags(tags, limit)", recall_tags_test)

        # recall_by_date()
        def recall_date_test():
            now = datetime.now(timezone.utc)
            start = now.replace(hour=0, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
            end = now.strftime('%Y-%m-%dT%H:%M:%SZ')
            results = mem.recall_by_date(start=start, end=end, limit=10)
            return f"Found {len(results)} in date range"
        test("recall_by_date(start, end, limit)", recall_date_test)

        # list_memories()
        def list_test():
            results = mem.list_memories(limit=50)
            return f"Listed {len(results)} memories"
        test("list_memories(limit)", list_test)

        # list_memories() with type filter
        def list_type_test():
            results = mem.list_memories(limit=50, memory_type="Learning")
            return f"Listed {len(results)} Learning memories"
        test("list_memories(memory_type='Learning')", list_type_test)

        # get_memory()
        def get_memory_test():
            result = mem.get_memory(mem_id)
            return f"Got memory: {result['content'][:30]}..."
        test("get_memory(id)", get_memory_test)

        # get_stats()
        def stats_test():
            stats = mem.get_stats()
            return f"Total: {stats['total_memories']} memories"
        test("get_stats()", stats_test)

        # === Proactive Context API (NEW) ===
        print("\n--- Proactive Context API ---")

        def proactive_test():
            result = mem.proactive_context(
                context="Testing the new proactive context API",
                semantic_threshold=0.5,
                max_results=5,
                auto_ingest=True,
                recency_weight=0.2
            )
            return f"Surfaced {result['count']} memories, latency: {result['latency_ms']:.1f}ms"
        test("proactive_context(context, ...)", proactive_test)

        def proactive_no_ingest():
            result = mem.proactive_context(
                context="Query without ingesting",
                auto_ingest=False,
                max_results=3
            )
            return f"Surfaced {result['count']} (no ingest)"
        test("proactive_context(auto_ingest=False)", proactive_no_ingest)

        # === Index Health API (NEW) ===
        print("\n--- Index Health API ---")

        def verify_index_test():
            result = mem.verify_index()
            return f"Healthy: {result['is_healthy']}, Storage: {result['total_storage']}, Indexed: {result['total_indexed']}"
        test("verify_index()", verify_index_test)

        def repair_index_test():
            result = mem.repair_index()
            return f"Repaired: {result['repaired']}, Failed: {result['failed']}"
        test("repair_index()", repair_index_test)

        def index_health_test():
            result = mem.index_health()
            return f"Vectors: {result['total_vectors']}, Healthy: {result['healthy']}"
        test("index_health()", index_health_test)

        # === Introspection APIs ===
        print("\n--- Introspection APIs ---")

        def context_summary_test():
            result = mem.context_summary(max_items=5)
            keys = list(result.keys())
            return f"Keys: {keys}"
        test("context_summary(max_items)", context_summary_test)

        def brain_state_test():
            result = mem.brain_state(longterm_limit=50)
            return f"Working: {len(result.get('working_memory', []))}, Session: {len(result.get('session_memory', []))}"
        test("brain_state(longterm_limit)", brain_state_test)

        def graph_stats_test():
            result = mem.graph_stats()
            return f"Nodes: {result.get('node_count', 0)}, Edges: {result.get('edge_count', 0)}"
        test("graph_stats()", graph_stats_test)

        def consolidation_report_test():
            now = datetime.now(timezone.utc)
            yesterday = now.replace(hour=0, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
            result = mem.consolidation_report(since=yesterday)
            return f"Keys: {list(result.keys())[:4]}..."
        test("consolidation_report(since)", consolidation_report_test)

        def consolidation_events_test():
            now = datetime.now(timezone.utc)
            yesterday = now.replace(hour=0, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
            result = mem.consolidation_events(since=yesterday)
            return f"Events: {len(result)}"
        test("consolidation_events(since)", consolidation_events_test)

        # === Forget APIs ===
        print("\n--- Forget APIs ---")

        # Add some test memories for forget tests
        forget_id = mem.remember("Temporary memory to forget", memory_type="Context", tags=["temp"])

        def forget_test():
            mem.forget(forget_id)
            return f"Deleted ID: {forget_id[:8]}..."
        test("forget(id)", forget_test)

        # Add more for other forget tests
        mem.remember("Old memory simulation", memory_type="Context", tags=["old-test"])

        def forget_by_tags_test():
            count = mem.forget_by_tags(tags=["old-test"])
            return f"Deleted by tags: {count}"
        test("forget_by_tags(tags)", forget_by_tags_test)

        def forget_by_age_test():
            count = mem.forget_by_age(days=365)  # Won't delete anything recent
            return f"Deleted older than 365 days: {count}"
        test("forget_by_age(days)", forget_by_age_test)

        def forget_by_importance_test():
            count = mem.forget_by_importance(threshold=0.01)  # Very low threshold
            return f"Deleted low importance: {count}"
        test("forget_by_importance(threshold)", forget_by_importance_test)

        def forget_by_pattern_test():
            mem.remember("DELETEME_pattern_test", memory_type="Context")
            count = mem.forget_by_pattern(pattern="DELETEME.*")
            return f"Deleted by pattern: {count}"
        test("forget_by_pattern(pattern)", forget_by_pattern_test)

        def forget_by_date_test():
            # Delete memories from last year (won't affect current test data)
            count = mem.forget_by_date(
                start="2024-01-01T00:00:00Z",
                end="2024-12-31T23:59:59Z"
            )
            return f"Deleted by date range: {count}"
        test("forget_by_date(start, end)", forget_by_date_test)

        # === Flush ===
        print("\n--- Storage ---")

        def flush_test():
            mem.flush()
            return "Flushed to disk"
        test("flush()", flush_test)

        # Final stats
        print("\n--- Final Verification ---")
        final_stats = mem.get_stats()
        test("Final stats check", lambda: f"Total memories: {final_stats['total_memories']}")

    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"\nCleaned up: {test_dir}")
        except:
            pass

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0

if __name__ == "__main__":
    success = test_all_apis()
    sys.exit(0 if success else 1)
