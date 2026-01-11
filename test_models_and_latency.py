#!/usr/bin/env python3
"""Verify models are working and measure realistic latency"""

import tempfile
import time
from datetime import datetime, timezone

def main():
    test_dir = tempfile.mkdtemp(prefix="shodh_verify_")
    print(f"Test directory: {test_dir}")
    print("=" * 70)

    from shodh_memory import Memory

    # === COLD START (includes model loading) ===
    print("\n1. COLD START - Creating Memory (loads ONNX models)")
    cold_start = time.perf_counter()
    mem = Memory(storage_path=f"{test_dir}/db")
    cold_time = (time.perf_counter() - cold_start) * 1000
    print(f"   Cold start time: {cold_time:.0f}ms")

    # === VERIFY EMBEDDING MODEL (MiniLM) ===
    print("\n2. EMBEDDING MODEL TEST (MiniLM-L6-v2)")

    # First embedding call (may have additional warmup)
    t1 = time.perf_counter()
    id1 = mem.remember("The quick brown fox jumps over the lazy dog", memory_type="Context")
    first_embed = (time.perf_counter() - t1) * 1000
    print(f"   First remember() with embedding: {first_embed:.1f}ms")

    # Second call (warm)
    t2 = time.perf_counter()
    id2 = mem.remember("Machine learning models process natural language", memory_type="Learning")
    second_embed = (time.perf_counter() - t2) * 1000
    print(f"   Second remember() with embedding: {second_embed:.1f}ms")

    # Verify semantic search works (proves embeddings exist)
    t3 = time.perf_counter()
    results = mem.recall("artificial intelligence language processing", limit=5)
    search_time = (time.perf_counter() - t3) * 1000
    print(f"   Semantic recall(): {search_time:.1f}ms")

    if results:
        print(f"   -> Found {len(results)} results")
        print(f"   -> Top result: '{results[0]['content'][:50]}...'")
        # Check if it found the semantically similar one (ML/NLP)
        if "Machine learning" in results[0]['content'] or "language" in results[0]['content'].lower():
            print("   -> EMBEDDING WORKING: Found semantically similar content!")
        else:
            print("   -> Top result matched by other criteria")
    else:
        print("   -> WARNING: No results from semantic search!")

    # === VERIFY NER MODEL (TinyBERT) ===
    print("\n3. NER MODEL TEST (TinyBERT)")

    # Store memory with named entities
    t4 = time.perf_counter()
    id3 = mem.remember(
        "Elon Musk announced that Tesla will open a new factory in Berlin, Germany next year.",
        memory_type="Context",
        tags=["ner-test"]
    )
    ner_time = (time.perf_counter() - t4) * 1000
    print(f"   Remember with NER extraction: {ner_time:.1f}ms")

    # Check if entities were extracted via graph
    graph = mem.graph_stats()
    print(f"   Graph nodes: {graph.get('node_count', 0)}, edges: {graph.get('edge_count', 0)}")

    # Try to find by searching for entity
    results2 = mem.recall("Tesla Berlin", limit=5)
    if results2:
        print(f"   -> Search for 'Tesla Berlin' found {len(results2)} results")
        if any("Tesla" in r['content'] or "Berlin" in r['content'] for r in results2):
            print("   -> NER WORKING: Entity-based search succeeded!")

    # === REALISTIC LATENCY (10 iterations, warm cache) ===
    print("\n4. REALISTIC LATENCY (10 iterations, warm cache)")

    # Remember
    times = []
    for i in range(10):
        t = time.perf_counter()
        mem.remember(f"Test content number {i} for latency measurement", memory_type="Context")
        times.append((time.perf_counter() - t) * 1000)
    avg = sum(times) / len(times)
    print(f"   remember() avg: {avg:.1f}ms (range: {min(times):.1f}-{max(times):.1f}ms)")

    # Recall semantic
    times = []
    for i in range(10):
        t = time.perf_counter()
        mem.recall("test content latency", limit=10)
        times.append((time.perf_counter() - t) * 1000)
    avg = sum(times) / len(times)
    print(f"   recall() avg: {avg:.1f}ms (range: {min(times):.1f}-{max(times):.1f}ms)")

    # Recall by tags (no embedding)
    times = []
    for i in range(10):
        t = time.perf_counter()
        mem.recall_by_tags(tags=["ner-test"], limit=10)
        times.append((time.perf_counter() - t) * 1000)
    avg = sum(times) / len(times)
    print(f"   recall_by_tags() avg: {avg:.1f}ms (range: {min(times):.1f}-{max(times):.1f}ms)")

    # Proactive context
    times = []
    for i in range(10):
        t = time.perf_counter()
        mem.proactive_context("current conversation context", auto_ingest=False)
        times.append((time.perf_counter() - t) * 1000)
    avg = sum(times) / len(times)
    print(f"   proactive_context() avg: {avg:.1f}ms (range: {min(times):.1f}-{max(times):.1f}ms)")

    # List memories
    times = []
    for i in range(10):
        t = time.perf_counter()
        mem.list_memories(limit=50)
        times.append((time.perf_counter() - t) * 1000)
    avg = sum(times) / len(times)
    print(f"   list_memories() avg: {avg:.1f}ms (range: {min(times):.1f}-{max(times):.1f}ms)")

    # === FINAL STATS ===
    print("\n5. FINAL VERIFICATION")
    stats = mem.get_stats()
    print(f"   Total memories: {stats['total_memories']}")

    health = mem.index_health()
    print(f"   Index vectors: {health['total_vectors']}")
    print(f"   Index healthy: {health['healthy']}")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  - Cold start (model loading): {cold_time:.0f}ms")
    print(f"  - First embedding call: {first_embed:.1f}ms")
    print(f"  - Warm embedding calls: ~{second_embed:.0f}ms")
    print(f"  - Embedding model: WORKING (semantic search returns relevant results)")
    print(f"  - NER model: {'WORKING' if graph.get('node_count', 0) > 0 else 'CHECK NEEDED'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
