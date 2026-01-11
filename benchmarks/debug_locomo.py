#!/usr/bin/env python3
"""Debug script to see what context is being retrieved for LOCOMO questions."""

import json
import os
import requests

# Load one item from dataset
mc10_path = os.path.expanduser('~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/data/locomo_mc10.json')

with open(mc10_path, 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]

item = items[0]  # First question

print("=" * 60)
print("QUESTION:", item["question"])
print("=" * 60)
print("\nQUESTION TYPE:", item["question_type"])
print("\nCHOICES:")
for i, c in enumerate(item["choices"]):
    marker = " <-- CORRECT" if i == item["correct_choice_index"] else ""
    print(f"  {i}. {c}{marker}")

print("\n" + "=" * 60)
print("CONVERSATION SESSIONS:")
print("=" * 60)

sessions = item.get("haystack_sessions", [])
summaries = item.get("haystack_session_summaries", [])

for i, session in enumerate(sessions[:2]):  # First 2 sessions
    print(f"\n--- Session {i+1} ---")
    summary = summaries[i] if i < len(summaries) else ""
    if summary:
        print(f"Summary: {summary[:200]}...")
    if session:
        for turn in session[:5]:  # First 5 turns
            if isinstance(turn, dict):
                speaker = turn.get("speaker", turn.get("role", "?"))
                content = turn.get("content", "")[:100]
                print(f"  {speaker}: {content}...")

# Now store and recall
print("\n" + "=" * 60)
print("STORING MEMORIES...")
print("=" * 60)

BASE_URL = "http://127.0.0.1:3030"
API_KEY = "sk-shodh-dev-local-testing-key"
USER_ID = "debug_locomo_test"

session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
})

# Store memories
batch_memories = []
for i, sess in enumerate(sessions):
    summary = summaries[i] if i < len(summaries) else ""
    if summary:
        batch_memories.append({
            "content": f"[Day {i+1}] Session {i+1} Summary: {summary}",
            "memory_type": "Context",
            "tags": [f"session_{i+1}", "summary"]
        })

    if sess:
        current_chunk = []
        current_len = 0
        for turn in sess:
            if isinstance(turn, dict):
                speaker = turn.get("speaker", turn.get("role", "Unknown"))
                content = turn.get("content", "")
                turn_text = f"{speaker}: {content}"
                turn_len = len(turn_text)

                if current_len + turn_len > 500 and current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    batch_memories.append({
                        "content": f"[Day {i+1}] Session {i+1}:\n{chunk_text}",
                        "memory_type": "Conversation",
                        "tags": [f"session_{i+1}"]
                    })
                    current_chunk = []
                    current_len = 0

                current_chunk.append(turn_text)
                current_len += turn_len

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            batch_memories.append({
                "content": f"[Day {i+1}] Session {i+1}:\n{chunk_text}",
                "memory_type": "Conversation",
                "tags": [f"session_{i+1}"]
            })

print(f"Storing {len(batch_memories)} memory chunks...")

payload = {
    "user_id": USER_ID,
    "memories": batch_memories,
    "options": {
        "extract_entities": True,
        "create_edges": True
    }
}

resp = session.post(f"{BASE_URL}/api/remember/batch", json=payload)
result = resp.json()
print(f"Stored: {result.get('created', 0)} memories")

# Recall
print("\n" + "=" * 60)
print("RECALLING CONTEXT FOR QUESTION...")
print("=" * 60)

query = item["question"]
print(f"\nQuery: {query}")

recall_payload = {
    "user_id": USER_ID,
    "query": query,
    "limit": 10,
    "mode": "associative"
}

resp = session.post(f"{BASE_URL}/api/recall", json=recall_payload)
memories = resp.json().get("memories", [])

print(f"\nRetrieved {len(memories)} memories:\n")

for i, mem in enumerate(memories):
    score = mem.get("score", 0)
    content = mem.get("content", "")
    print(f"[{i+1}] Score: {score:.3f}")
    print(f"    {content[:400]}")
    print()

# Check if the answer is in the retrieved context
print("=" * 60)
print("ANSWER ANALYSIS")
print("=" * 60)
correct_answer = item["choices"][item["correct_choice_index"]]
print(f"\nCorrect answer: {correct_answer}")
print("\nIs answer mentioned in retrieved context?")

all_context = " ".join([m.get("content", "") for m in memories]).lower()
answer_words = correct_answer.lower().split()
matches = sum(1 for w in answer_words if len(w) > 3 and w in all_context)
print(f"  Word overlap: {matches}/{len([w for w in answer_words if len(w) > 3])}")

# Cleanup
print("\n[Cleaning up test data...]")
session.delete(f"{BASE_URL}/api/users/{USER_ID}", params={"purge": "true"})
print("Done!")
