#!/usr/bin/env python3
"""Debug script to visualize graph structure for LOCOMO questions."""

import json
import os
import requests

BASE_URL = "http://127.0.0.1:3030"
API_KEY = "sk-shodh-dev-local-testing-key"
USER_ID = "debug_graph_test"

session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
})

# Load one item from dataset
mc10_path = os.path.expanduser('~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/data/locomo_mc10.json')

with open(mc10_path, 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]

item = items[0]  # First question (multi-hop)

print("=" * 70)
print("QUESTION:", item["question"])
print("TYPE:", item["question_type"])
print("=" * 70)
print("\nCHOICES:")
for i, c in enumerate(item["choices"]):
    marker = " <-- CORRECT" if i == item["correct_choice_index"] else ""
    print(f"  {i}. {c}{marker}")

# Store conversations
print("\n" + "=" * 70)
print("STORING MEMORIES...")
print("=" * 70)

sessions = item.get("haystack_sessions", [])
summaries = item.get("haystack_session_summaries", [])

stored_count = 0
for i, summary in enumerate(summaries):
    sess = sessions[i] if i < len(sessions) else []

    # Store summary
    if summary and isinstance(summary, str) and summary.strip():
        resp = session.post(f"{BASE_URL}/api/remember", json={
            "user_id": USER_ID,
            "content": f"Session {i+1} Summary: {summary[:2000]}",
            "memory_type": "Context",
            "tags": [f"session_{i+1}", "summary"]
        })
        if resp.status_code == 200:
            stored_count += 1

    # Store dialogue chunks
    if sess and isinstance(sess, list):
        dialogue_text = ""
        for turn in sess:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                dialogue_text += content + "\n"

        if dialogue_text.strip():
            chunks = [dialogue_text[j:j+500] for j in range(0, len(dialogue_text), 500)]
            for chunk in chunks[:10]:
                resp = session.post(f"{BASE_URL}/api/remember", json={
                    "user_id": USER_ID,
                    "content": f"Session {i+1}: {chunk}",
                    "memory_type": "Conversation",
                    "tags": [f"session_{i+1}", "dialogue"]
                })
                if resp.status_code == 200:
                    stored_count += 1

print(f"Stored {stored_count} memories")

# Check graph stats
print("\n" + "=" * 70)
print("GRAPH STATISTICS")
print("=" * 70)

resp = session.get(f"{BASE_URL}/api/graph/{USER_ID}/stats")
if resp.status_code == 200:
    stats = resp.json()
    print(f"Entities: {stats.get('entity_count', 0)}")
    print(f"Relationships: {stats.get('relationship_count', 0)}")
    print(f"Episodes: {stats.get('episode_count', 0)}")
else:
    print(f"ERROR: {resp.status_code} - {resp.text}")

# Get all entities
print("\n" + "=" * 70)
print("ENTITIES IN GRAPH (first 30)")
print("=" * 70)

resp = session.post(f"{BASE_URL}/api/graph/entities/all", json={
    "user_id": USER_ID,
    "limit": 30
})
if resp.status_code == 200:
    data = resp.json()
    entities = data.get("entities", [])
    for ent in entities:
        print(f"  - {ent.get('name', '?')} (mentions: {ent.get('mention_count', 0)}, salience: {ent.get('salience', 0):.2f})")

# Query analysis
print("\n" + "=" * 70)
print("QUERY ANALYSIS")
print("=" * 70)

question = item["question"]
print(f"Query: {question}")

# Extract key terms manually for debugging
words = question.lower().split()
key_terms = [w.strip('?.,!') for w in words if len(w) > 3 and w not in ['what', 'when', 'where', 'which', 'that', 'this', 'with', 'from', 'have', 'does', 'did']]
print(f"Key terms: {key_terms}")

# Check which entities from query exist in graph
print("\n" + "=" * 70)
print("ENTITY LOOKUP FOR QUERY")
print("=" * 70)

for term in key_terms[:5]:
    # Try to find entity - endpoint returns entity directly or null
    resp = session.post(f"{BASE_URL}/api/graph/entity/find", json={
        "user_id": USER_ID,
        "entity_name": term
    })
    if resp.status_code == 200:
        ent = resp.json()
        if ent:
            print(f"  '{term}' -> FOUND: {ent.get('name')} (uuid: {ent.get('uuid', '')[:8]}...)")
        else:
            print(f"  '{term}' -> NOT FOUND")
    else:
        print(f"  '{term}' -> ERROR: {resp.status_code}")

# Recall with graph mode
print("\n" + "=" * 70)
print("RECALL RESULTS (mode=hybrid)")
print("=" * 70)

resp = session.post(f"{BASE_URL}/api/recall", json={
    "user_id": USER_ID,
    "query": question,
    "limit": 10,
    "mode": "hybrid"
})
if resp.status_code == 200:
    memories = resp.json().get("memories", [])
    print(f"Retrieved {len(memories)} memories:\n")
    for i, mem in enumerate(memories):
        score = mem.get("score", 0)
        content = mem.get("experience", {}).get("content", "")[:200]
        print(f"[{i+1}] Score: {score:.4f}")
        print(f"    {content}...")
        print()

# Cleanup
print("\n[Cleaning up test data...]")
session.delete(f"{BASE_URL}/api/users/{USER_ID}", params={"purge": "true"})
print("Done!")
