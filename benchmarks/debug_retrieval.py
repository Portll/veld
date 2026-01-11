#!/usr/bin/env python3
"""Debug what memories are retrieved for failing LOCOMO questions."""

import json
import os
import requests
import time

BASE_URL = "http://127.0.0.1:3030"
API_KEY = "sk-shodh-dev-local-testing-key"

session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
})

# Load dataset
mc10_path = os.path.expanduser('~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/data/locomo_mc10.json')

with open(mc10_path, 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f][:10]

# Failing questions from last run
FAILING_QS = [5, 6, 9]  # q5, q6, q9 (multi-hop failures)

for q_idx in FAILING_QS:
    item = items[q_idx]
    user_id = f"debug_q{q_idx}"

    print("=" * 80)
    print(f"Q{q_idx}: {item['question']}")
    print(f"Type: {item['question_type']}")
    print(f"Correct answer: {item['choices'][item['correct_choice_index']]}")
    print("=" * 80)

    # Clean up any existing data
    session.delete(f"{BASE_URL}/api/users/{user_id}", params={"purge": "true"})

    # Store conversations (same as benchmark)
    sessions = item.get("haystack_sessions", [])
    summaries = item.get("haystack_session_summaries", [])

    stored_count = 0
    for i, summary in enumerate(summaries):
        sess = sessions[i] if i < len(sessions) else []

        # Store summary
        if summary and isinstance(summary, str) and summary.strip():
            resp = session.post(f"{BASE_URL}/api/remember", json={
                "user_id": user_id,
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
                for chunk_idx, chunk in enumerate(chunks[:10]):
                    resp = session.post(f"{BASE_URL}/api/remember", json={
                        "user_id": user_id,
                        "content": f"Session {i+1}: {chunk}",
                        "memory_type": "Conversation",
                        "tags": [f"session_{i+1}", "dialogue", f"chunk_{chunk_idx}"]
                    })
                    if resp.status_code == 200:
                        stored_count += 1

    print(f"\nStored {stored_count} memories")

    # Small delay for indexing
    time.sleep(0.5)

    # Recall for the question
    print(f"\n--- RETRIEVED MEMORIES (limit=5) ---")
    resp = session.post(f"{BASE_URL}/api/recall", json={
        "user_id": user_id,
        "query": item["question"],
        "limit": 5,
        "mode": "hybrid"
    })

    if resp.status_code == 200:
        memories = resp.json().get("memories", [])
        print(f"Retrieved {len(memories)} memories:\n")

        for i, mem in enumerate(memories):
            score = mem.get("score", 0)
            content = mem.get("experience", {}).get("content", "")
            tags = mem.get("experience", {}).get("tags", [])

            print(f"[{i+1}] Score: {score:.4f} | Tags: {tags}")
            print(f"    Content preview: {content[:200]}...")
            print()

    # Also search for the correct answer text in stored memories
    correct_answer = item['choices'][item['correct_choice_index']]
    print(f"\n--- SEARCHING FOR CORRECT ANSWER: '{correct_answer}' ---")

    # Get all memories and search
    resp = session.post(f"{BASE_URL}/api/memories", json={
        "user_id": user_id,
        "limit": 200
    })

    if resp.status_code == 200:
        all_memories = resp.json().get("memories", [])

        # Search for answer text in memories
        matches = []
        for mem in all_memories:
            content = mem.get("experience", {}).get("content", "").lower()
            if any(word.lower() in content for word in correct_answer.split() if len(word) > 3):
                matches.append(mem)

        if matches:
            print(f"Found {len(matches)} memories containing answer keywords:\n")
            for mem in matches[:3]:
                content = mem.get("experience", {}).get("content", "")
                tags = mem.get("experience", {}).get("tags", [])
                print(f"  Tags: {tags}")
                print(f"  Content: {content[:300]}...")
                print()
        else:
            print("  NO memories contain the answer keywords!")
            print(f"  Answer words searched: {[w for w in correct_answer.split() if len(w) > 3]}")

    # Cleanup
    session.delete(f"{BASE_URL}/api/users/{user_id}", params={"purge": "true"})
    print("\n")
