#!/usr/bin/env python3
"""
Comprehensive API test for shodh-memory 0.1.61

Tests ALL Python API endpoints based on src/python.rs bindings.
"""
import sys
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

def main():
    print('='*60)
    print('SHODH-MEMORY 0.1.61 - COMPREHENSIVE API TEST')
    print('='*60)

    # Clean test directory
    test_dir = './test_wheel_api'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir, exist_ok=True)

    passed = 0
    failed = 0

    # =========================================================================
    # [1] IMPORT TEST
    # =========================================================================
    print('\n[1] IMPORT TEST')
    print('-'*40)
    try:
        from shodh_memory import (
            Memory, MemorySystem,
            Position, GeoLocation, GeoFilter,
            DecisionContext, Outcome, Environment
        )
        print('[OK] All classes imported successfully')
        print(f'     Memory == MemorySystem: {Memory is MemorySystem}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Import failed: {e}')
        failed += 1
        return 1

    # =========================================================================
    # [2] INITIALIZATION TEST
    # =========================================================================
    print('\n[2] INITIALIZATION TEST')
    print('-'*40)
    try:
        start = time.time()
        memory = Memory(test_dir, robot_id="test-robot-001")
        elapsed = time.time() - start
        print(f'[OK] Memory initialized in {elapsed:.2f}s')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Init failed: {e}')
        failed += 1
        return 1

    # =========================================================================
    # [3] REMEMBER API - Core memory storage
    # =========================================================================
    print('\n[3] REMEMBER API TEST')
    print('-'*40)
    test_memories = [
        {'content': 'Python is a programming language', 'memory_type': 'observation', 'tags': ['python', 'programming']},
        {'content': 'Rust is fast and memory-safe', 'memory_type': 'learning', 'tags': ['rust', 'performance']},
        {'content': 'Machine learning uses neural networks', 'memory_type': 'discovery', 'tags': ['ml', 'ai']},
        {'content': 'Docker containers simplify deployment', 'memory_type': 'context', 'tags': ['docker', 'devops']},
        {'content': 'GraphQL is an alternative to REST', 'memory_type': 'pattern', 'tags': ['api', 'graphql']},
    ]

    memory_ids = []
    for i, mem in enumerate(test_memories):
        try:
            start = time.time()
            mid = memory.remember(
                content=mem['content'],
                memory_type=mem['memory_type'],
                tags=mem['tags']
            )
            elapsed = time.time() - start
            memory_ids.append(mid)
            print(f'[OK] Memory {i+1} stored: {mid[:8]}... ({elapsed:.3f}s)')
            passed += 1
        except Exception as e:
            print(f'[FAIL] Remember failed for memory {i+1}: {e}')
            failed += 1

    print(f'\nTotal memories stored: {len(memory_ids)}')

    # =========================================================================
    # [4] RECALL (SEMANTIC SEARCH) TEST
    # =========================================================================
    print('\n[4] RECALL (SEMANTIC SEARCH) TEST')
    print('-'*40)
    queries = [
        ('programming languages', 'hybrid'),
        ('performance and speed', 'semantic'),
        ('artificial intelligence', 'similarity'),
    ]

    for query, mode in queries:
        try:
            start = time.time()
            results = memory.recall(query=query, limit=3, mode=mode)
            elapsed = time.time() - start
            print(f'[OK] Query "{query}" ({mode}): {len(results)} results ({elapsed:.3f}s)')
            for r in results[:2]:
                content = r.get('content', 'N/A')[:40] if isinstance(r, dict) else str(r)[:40]
                print(f'     - {content}...')
            passed += 1
        except Exception as e:
            print(f'[FAIL] Recall failed for "{query}": {e}')
            failed += 1

    # =========================================================================
    # [5] RECALL BY TAGS TEST
    # =========================================================================
    print('\n[5] RECALL BY TAGS TEST')
    print('-'*40)
    tag_tests = [['python'], ['rust', 'performance'], ['ml', 'ai']]
    for tags in tag_tests:
        try:
            start = time.time()
            results = memory.recall_by_tags(tags=tags, limit=5)
            elapsed = time.time() - start
            print(f'[OK] Tags {tags}: {len(results)} results ({elapsed:.3f}s)')
            passed += 1
        except Exception as e:
            print(f'[FAIL] Recall by tags failed for {tags}: {e}')
            failed += 1

    # =========================================================================
    # [6] RECALL BY DATE TEST (RFC3339 format required)
    # =========================================================================
    print('\n[6] RECALL BY DATE TEST')
    print('-'*40)
    try:
        # RFC3339 format: 2024-01-01T00:00:00Z
        start_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        start = time.time()
        results = memory.recall_by_date(start=start_date, end=end_date, limit=10)
        elapsed = time.time() - start
        print(f'[OK] Date range query: {len(results)} results ({elapsed:.3f}s)')
        print(f'     Range: {start_date} to {end_date}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Recall by date failed: {e}')
        failed += 1

    # =========================================================================
    # [7] LIST MEMORIES TEST
    # =========================================================================
    print('\n[7] LIST MEMORIES TEST')
    print('-'*40)
    try:
        start = time.time()
        all_mems = memory.list_memories(limit=10)
        elapsed = time.time() - start
        print(f'[OK] Listed {len(all_mems)} memories ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] List memories failed: {e}')
        failed += 1

    # =========================================================================
    # [8] GET_STATS TEST
    # =========================================================================
    print('\n[8] GET_STATS TEST')
    print('-'*40)
    try:
        start = time.time()
        stats = memory.get_stats()
        elapsed = time.time() - start
        print(f'[OK] Stats retrieved ({elapsed:.3f}s)')
        for k, v in stats.items():
            print(f'     {k}: {v}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] get_stats failed: {e}')
        failed += 1

    # =========================================================================
    # [9] GRAPH_STATS TEST
    # =========================================================================
    print('\n[9] GRAPH_STATS TEST')
    print('-'*40)
    try:
        start = time.time()
        gstats = memory.graph_stats()
        elapsed = time.time() - start
        print(f'[OK] Graph stats retrieved ({elapsed:.3f}s)')
        for k, v in gstats.items():
            print(f'     {k}: {v}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] graph_stats failed: {e}')
        failed += 1

    # =========================================================================
    # [10] CONTEXT SUMMARY TEST
    # =========================================================================
    print('\n[10] CONTEXT SUMMARY TEST')
    print('-'*40)
    try:
        start = time.time()
        summary = memory.context_summary(max_items=5)
        elapsed = time.time() - start
        print(f'[OK] Context summary retrieved ({elapsed:.3f}s)')
        print(f'     Total memories: {summary.get("total_memories", "N/A")}')
        print(f'     Learnings: {len(summary.get("learnings", []))}')
        print(f'     Decisions: {len(summary.get("decisions", []))}')
        print(f'     Context: {len(summary.get("context", []))}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Context summary failed: {e}')
        failed += 1

    # =========================================================================
    # [11] CONSOLIDATION REPORT TEST
    # =========================================================================
    print('\n[11] CONSOLIDATION REPORT TEST')
    print('-'*40)
    try:
        start = time.time()
        report = memory.consolidation_report()
        elapsed = time.time() - start
        print(f'[OK] Consolidation report retrieved ({elapsed:.3f}s)')
        if 'stats' in report:
            print(f'     Memories strengthened: {report["stats"].get("memories_strengthened", 0)}')
            print(f'     Edges formed: {report["stats"].get("edges_formed", 0)}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Consolidation report failed: {e}')
        failed += 1

    # =========================================================================
    # [12] CONSOLIDATION EVENTS TEST
    # =========================================================================
    print('\n[12] CONSOLIDATION EVENTS TEST')
    print('-'*40)
    try:
        start = time.time()
        events = memory.consolidation_events()
        elapsed = time.time() - start
        print(f'[OK] Consolidation events: {len(events)} events ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Consolidation events failed: {e}')
        failed += 1

    # =========================================================================
    # [13] BRAIN STATE TEST
    # =========================================================================
    print('\n[13] BRAIN STATE TEST')
    print('-'*40)
    try:
        start = time.time()
        state = memory.brain_state()
        elapsed = time.time() - start
        print(f'[OK] Brain state retrieved ({elapsed:.3f}s)')
        print(f'     Working: {len(state.get("working_memory", []))} memories')
        print(f'     Session: {len(state.get("session_memory", []))} memories')
        print(f'     Long-term: {len(state.get("longterm_memory", []))} memories')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Brain state failed: {e}')
        failed += 1

    # =========================================================================
    # [14] GET MEMORY BY ID TEST
    # =========================================================================
    print('\n[14] GET MEMORY BY ID TEST')
    print('-'*40)
    if memory_ids:
        try:
            mid = memory_ids[0]
            start = time.time()
            mem = memory.get_memory(mid)
            elapsed = time.time() - start
            print(f'[OK] Got memory {mid[:8]}... ({elapsed:.3f}s)')
            print(f'     Content: {mem.get("content", "N/A")[:50]}...')
            passed += 1
        except Exception as e:
            print(f'[FAIL] Get memory failed: {e}')
            failed += 1

    # =========================================================================
    # [15] FORGET TEST
    # =========================================================================
    print('\n[15] FORGET TEST')
    print('-'*40)
    if memory_ids:
        try:
            mid_to_delete = memory_ids.pop()  # Remove last one
            start = time.time()
            result = memory.forget(mid_to_delete)
            elapsed = time.time() - start
            print(f'[OK] Memory {mid_to_delete[:8]}... deleted: {result} ({elapsed:.3f}s)')
            remaining = memory.list_memories(limit=100)
            print(f'     Remaining memories: {len(remaining)}')
            passed += 1
        except Exception as e:
            print(f'[FAIL] Forget failed: {e}')
            failed += 1

    # =========================================================================
    # [16] FORGET BY TAGS TEST
    # =========================================================================
    print('\n[16] FORGET BY TAGS TEST')
    print('-'*40)
    try:
        start = time.time()
        count = memory.forget_by_tags(tags=['docker'])
        elapsed = time.time() - start
        print(f'[OK] Forgot by tags [docker]: {count} memories ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Forget by tags failed: {e}')
        failed += 1

    # =========================================================================
    # [17] MISSION API TEST
    # =========================================================================
    print('\n[17] ROBOTICS: MISSION API TEST')
    print('-'*40)
    try:
        memory.start_mission('test-mission-001')
        current = memory.current_mission()
        print(f'[OK] Mission started: {current}')
        memory.end_mission()
        after = memory.current_mission()
        print(f'[OK] Mission ended. Current: {after}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Mission API failed: {e}')
        failed += 1

    # =========================================================================
    # [18] RECORD DECISION TEST (requires DecisionContext and Outcome)
    # =========================================================================
    print('\n[18] ROBOTICS: RECORD DECISION TEST')
    print('-'*40)
    try:
        ctx = DecisionContext(
            state={'battery': '85', 'obstacle_ahead': 'true'},
            action_params={'speed': '0.5'},
            confidence=0.85,
            alternatives=['turn_right', 'stop']
        )
        outcome = Outcome(
            outcome_type='success',
            details='Successfully avoided obstacle',
            reward=1.0
        )
        start = time.time()
        mid = memory.record_decision(
            description='Decided to turn left to avoid obstacle',
            action_type='turn_left',
            decision_context=ctx,
            outcome=outcome
        )
        elapsed = time.time() - start
        print(f'[OK] Decision recorded: {mid[:8]}... ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Record decision failed: {e}')
        failed += 1

    # =========================================================================
    # [19] RECORD FAILURE TEST
    # =========================================================================
    print('\n[19] ROBOTICS: RECORD FAILURE TEST')
    print('-'*40)
    try:
        start = time.time()
        mid = memory.record_failure(
            description='Motor overheated during climb',
            severity='warning',
            root_cause='excessive load',
            recovery_action='cooldown and retry'
        )
        elapsed = time.time() - start
        print(f'[OK] Failure recorded: {mid[:8]}... ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Record failure failed: {e}')
        failed += 1

    # =========================================================================
    # [20] FIND FAILURES TEST
    # =========================================================================
    print('\n[20] ROBOTICS: FIND FAILURES TEST')
    print('-'*40)
    try:
        start = time.time()
        failures = memory.find_failures(max_results=5)
        elapsed = time.time() - start
        print(f'[OK] Found {len(failures)} failures ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Find failures failed: {e}')
        failed += 1

    # =========================================================================
    # [21] RECORD WAYPOINT TEST
    # =========================================================================
    print('\n[21] ROBOTICS: RECORD WAYPOINT TEST')
    print('-'*40)
    try:
        pos = Position(x=10.5, y=20.3, z=5.0)
        geo = GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=50.0)
        start = time.time()
        mid = memory.record_waypoint(
            waypoint_id='checkpoint-alpha',
            status='reached',
            position=pos,
            geo_location=geo
        )
        elapsed = time.time() - start
        print(f'[OK] Waypoint recorded: {mid[:8]}... ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Record waypoint failed: {e}')
        failed += 1

    # =========================================================================
    # [22] RECORD SENSOR TEST
    # =========================================================================
    print('\n[22] ROBOTICS: RECORD SENSOR TEST')
    print('-'*40)
    try:
        start = time.time()
        mid = memory.record_sensor(
            sensor_name='thermal',
            readings={'temp': 45.5, 'humidity': 65.0}
        )
        elapsed = time.time() - start
        print(f'[OK] Sensor recorded: {mid[:8]}... ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Record sensor failed: {e}')
        failed += 1

    # =========================================================================
    # [23] RECORD OBSTACLE TEST
    # =========================================================================
    print('\n[23] ROBOTICS: RECORD OBSTACLE TEST')
    print('-'*40)
    try:
        start = time.time()
        mid = memory.record_obstacle(
            description='Large rock detected ahead',
            distance=2.5,
            confidence=0.95
        )
        elapsed = time.time() - start
        print(f'[OK] Obstacle recorded: {mid[:8]}... ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Record obstacle failed: {e}')
        failed += 1

    # =========================================================================
    # [24] RECORD ANOMALY TEST (requires sensor_data)
    # =========================================================================
    print('\n[24] ROBOTICS: RECORD ANOMALY TEST')
    print('-'*40)
    try:
        start = time.time()
        mid = memory.record_anomaly(
            description='Unexpected vibration pattern detected',
            sensor_data={'vibration_x': 0.8, 'vibration_y': 1.2, 'vibration_z': 0.3},
            severity='info'
        )
        elapsed = time.time() - start
        print(f'[OK] Anomaly recorded: {mid[:8]}... ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Record anomaly failed: {e}')
        failed += 1

    # =========================================================================
    # [25] FIND ANOMALIES TEST
    # =========================================================================
    print('\n[25] ROBOTICS: FIND ANOMALIES TEST')
    print('-'*40)
    try:
        start = time.time()
        anomalies = memory.find_anomalies(max_results=5)
        elapsed = time.time() - start
        print(f'[OK] Found {len(anomalies)} anomalies ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Find anomalies failed: {e}')
        failed += 1

    # =========================================================================
    # [26] FIND SIMILAR DECISIONS TEST
    # =========================================================================
    print('\n[26] ROBOTICS: FIND SIMILAR DECISIONS TEST')
    print('-'*40)
    try:
        start = time.time()
        decisions = memory.find_similar_decisions(action_type='turn_left', max_results=5)
        elapsed = time.time() - start
        print(f'[OK] Found {len(decisions)} similar decisions ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Find similar decisions failed: {e}')
        failed += 1

    # =========================================================================
    # [27] FIND BY PATTERN TEST
    # =========================================================================
    print('\n[27] ROBOTICS: FIND BY PATTERN TEST')
    print('-'*40)
    try:
        start = time.time()
        matches = memory.find_by_pattern(pattern_id='test-pattern', max_results=5)
        elapsed = time.time() - start
        print(f'[OK] Pattern match: {len(matches)} results ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Find by pattern failed: {e}')
        failed += 1

    # =========================================================================
    # [28] FORGET BY AGE TEST
    # =========================================================================
    print('\n[28] FORGET BY AGE TEST')
    print('-'*40)
    try:
        start = time.time()
        count = memory.forget_by_age(days=365)  # Won't delete anything recent
        elapsed = time.time() - start
        print(f'[OK] Forgot by age (365 days): {count} memories ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Forget by age failed: {e}')
        failed += 1

    # =========================================================================
    # [29] FORGET BY IMPORTANCE TEST
    # =========================================================================
    print('\n[29] FORGET BY IMPORTANCE TEST')
    print('-'*40)
    try:
        start = time.time()
        count = memory.forget_by_importance(threshold=0.01)  # Very low threshold
        elapsed = time.time() - start
        print(f'[OK] Forgot by importance (<0.01): {count} memories ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Forget by importance failed: {e}')
        failed += 1

    # =========================================================================
    # [30] FORGET BY PATTERN TEST
    # =========================================================================
    print('\n[30] FORGET BY PATTERN TEST')
    print('-'*40)
    try:
        start = time.time()
        count = memory.forget_by_pattern(pattern='nonexistent_pattern_xyz')
        elapsed = time.time() - start
        print(f'[OK] Forgot by pattern: {count} memories ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Forget by pattern failed: {e}')
        failed += 1

    # =========================================================================
    # [31] FLUSH TEST
    # =========================================================================
    print('\n[31] FLUSH TEST')
    print('-'*40)
    try:
        start = time.time()
        memory.flush()
        elapsed = time.time() - start
        print(f'[OK] Flushed to disk ({elapsed:.3f}s)')
        passed += 1
    except Exception as e:
        print(f'[FAIL] Flush failed: {e}')
        failed += 1

    # =========================================================================
    # FINAL STATS
    # =========================================================================
    print('\n' + '='*60)
    print('FINAL STATS')
    print('='*60)
    try:
        final_stats = memory.get_stats()
        for k, v in final_stats.items():
            print(f'  {k}: {v}')
    except Exception as e:
        print(f'  Could not get final stats: {e}')

    # =========================================================================
    # TEST SUMMARY
    # =========================================================================
    print('\n' + '='*60)
    print('TEST SUMMARY')
    print('='*60)
    print(f'PASSED: {passed}')
    print(f'FAILED: {failed}')
    print(f'TOTAL:  {passed + failed}')

    # Cleanup - del memory first to release RocksDB locks
    del memory
    time.sleep(0.5)  # Give time for RocksDB to release locks

    try:
        shutil.rmtree(test_dir, ignore_errors=True)
        print('\nTest directory cleaned up')
    except Exception as e:
        print(f'\nWarning: Could not clean test directory: {e}')

    if failed == 0:
        print('\n=== ALL TESTS PASSED ===')
        return 0
    else:
        print(f'\n=== {failed} TESTS FAILED ===')
        return 1

if __name__ == '__main__':
    sys.exit(main())
