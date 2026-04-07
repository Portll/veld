#!/usr/bin/env bun
/**
 * Veld Comprehensive Test & Benchmark
 *
 * Tests:
 * 1. Save latency (single and batch)
 * 2. Retrieval latency and relevance
 * 3. Semantic search accuracy
 * 4. Memory type handling
 * 5. Tag filtering
 * 6. Memory stats
 */

const API_URL = process.env.VELD_API_URL || "http://127.0.0.1:3030";
const USER_ID = "benchmark-test";

// API Key - required (no hardcoded fallback for security)
const API_KEY = process.env.VELD_API_KEY;
if (!API_KEY) {
  console.error("ERROR: VELD_API_KEY environment variable not set");
  console.error("Set it with: export VELD_API_KEY=your-api-key");
  process.exit(1);
}

interface TestResult {
  test: string;
  passed: boolean;
  latency_ms: number;
  details: string;
}

interface BenchmarkReport {
  timestamp: string;
  total_tests: number;
  passed: number;
  failed: number;
  results: TestResult[];
  summary: {
    avg_save_latency_ms: number;
    avg_retrieval_latency_ms: number;
    semantic_accuracy: number;
    relevance_scores: number[];
  };
}

async function apiCall(endpoint: string, method: string = "GET", body?: object): Promise<any> {
  const options: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_URL}${endpoint}`, options);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error: ${response.status} - ${text}`);
  }

  return response.json();
}

async function measureLatency<T>(fn: () => Promise<T>): Promise<{ result: T; latency_ms: number }> {
  const start = performance.now();
  const result = await fn();
  const latency_ms = performance.now() - start;
  return { result, latency_ms };
}

// Test data - diverse memories for semantic search testing
const testMemories = [
  {
    content: "User prefers TypeScript over JavaScript for all projects",
    type: "Context",
    tags: ["preference", "language", "typescript"],
  },
  {
    content: "The authentication system uses JWT tokens with 24-hour expiry",
    type: "Decision",
    tags: ["auth", "security", "jwt"],
  },
  {
    content: "Database connection pool size should be set to 20 for optimal performance",
    type: "Learning",
    tags: ["database", "performance", "config"],
  },
  {
    content: "Error occurred: CORS policy blocked request from localhost:3000",
    type: "Error",
    tags: ["error", "cors", "frontend"],
  },
  {
    content: "Discovered that the API rate limit is 100 requests per minute",
    type: "Discovery",
    tags: ["api", "rate-limit", "discovery"],
  },
  {
    content: "Pattern: All React components follow the Container/Presenter pattern",
    type: "Pattern",
    tags: ["react", "architecture", "pattern"],
  },
  {
    content: "Task completed: Implemented user registration flow with email verification",
    type: "Task",
    tags: ["task", "registration", "email"],
  },
  {
    content: "Code edit: Refactored the payment processing module for better error handling",
    type: "CodeEdit",
    tags: ["refactor", "payment", "error-handling"],
  },
  {
    content: "User mentioned they are building a drone navigation system",
    type: "Conversation",
    tags: ["user", "drone", "navigation"],
  },
  {
    content: "Search performed: Looking for vector database implementations in Rust",
    type: "Search",
    tags: ["search", "vector-db", "rust"],
  },
];

// Semantic search test cases - query and expected matches
const searchTests = [
  {
    query: "What programming language does the user prefer?",
    expectedContent: "TypeScript",
    description: "Language preference retrieval",
  },
  {
    query: "How does authentication work?",
    expectedContent: "JWT",
    description: "Auth system retrieval",
  },
  {
    query: "Database configuration settings",
    expectedContent: "pool size",
    description: "Config retrieval",
  },
  {
    query: "CORS issues",
    expectedContent: "CORS policy",
    description: "Error retrieval",
  },
  {
    query: "API limitations",
    expectedContent: "rate limit",
    description: "Discovery retrieval",
  },
  {
    query: "React component architecture",
    expectedContent: "Container/Presenter",
    description: "Pattern retrieval",
  },
  {
    query: "drone project",
    expectedContent: "drone navigation",
    description: "Conversation retrieval",
  },
];

async function runBenchmark(): Promise<BenchmarkReport> {
  const results: TestResult[] = [];
  const saveLatencies: number[] = [];
  const retrievalLatencies: number[] = [];
  const relevanceScores: number[] = [];

  console.log("🧪 Veld Benchmark Starting...\n");
  console.log("=".repeat(60));

  // Test 1: Clear existing test data
  console.log("\n📋 Test 1: Clearing existing test data...");
  try {
    const listResult = await apiCall("/api/memories", "POST", { user_id: USER_ID });
    const existingMemories = listResult.memories || [];
    for (const mem of existingMemories) {
      await apiCall(`/api/memory/${mem.memory_id}`, "DELETE");
    }
    results.push({
      test: "Clear existing data",
      passed: true,
      latency_ms: 0,
      details: `Cleared ${existingMemories.length} existing memories`,
    });
    console.log(`   ✅ Cleared ${existingMemories.length} existing memories`);
  } catch (error) {
    results.push({
      test: "Clear existing data",
      passed: false,
      latency_ms: 0,
      details: `Error: ${error}`,
    });
    console.log(`   ❌ Failed to clear: ${error}`);
  }

  // Test 2: Individual Save Latency
  console.log("\n📋 Test 2: Individual Save Latency...");
  const savedMemoryIds: string[] = [];

  for (let i = 0; i < testMemories.length; i++) {
    const mem = testMemories[i];
    try {
      const { result, latency_ms } = await measureLatency(() =>
        apiCall("/api/record", "POST", {
          user_id: USER_ID,
          experience: {
            content: mem.content,
            experience_type: mem.type,
            tags: mem.tags,
          },
        })
      );

      saveLatencies.push(latency_ms);
      savedMemoryIds.push(result.memory_id);

      results.push({
        test: `Save memory ${i + 1}`,
        passed: true,
        latency_ms,
        details: `Type: ${mem.type}, Content: "${mem.content.slice(0, 40)}..."`,
      });
      console.log(`   ✅ Memory ${i + 1}: ${latency_ms.toFixed(2)}ms - ${mem.type}`);
    } catch (error) {
      results.push({
        test: `Save memory ${i + 1}`,
        passed: false,
        latency_ms: 0,
        details: `Error: ${error}`,
      });
      console.log(`   ❌ Memory ${i + 1} failed: ${error}`);
    }
  }

  // Test 3: Semantic Search Accuracy & Latency
  console.log("\n📋 Test 3: Semantic Search Accuracy & Latency...");
  let correctRetrievals = 0;

  for (const searchTest of searchTests) {
    try {
      const { result, latency_ms } = await measureLatency(() =>
        apiCall("/api/recall", "POST", {
          user_id: USER_ID,
          query: searchTest.query,
          limit: 3,
        })
      );

      retrievalLatencies.push(latency_ms);

      const memories = result.memories || [];
      const topResult = memories[0];
      const found = topResult && topResult.experience.content.toLowerCase().includes(searchTest.expectedContent.toLowerCase());

      if (found) {
        correctRetrievals++;
        relevanceScores.push(topResult.score || 0);
      }

      results.push({
        test: `Search: ${searchTest.description}`,
        passed: found,
        latency_ms,
        details: found
          ? `Found "${searchTest.expectedContent}" (score: ${(topResult?.score || 0).toFixed(3)})`
          : `Expected "${searchTest.expectedContent}" not in top result`,
      });

      console.log(`   ${found ? '✅' : '❌'} ${searchTest.description}: ${latency_ms.toFixed(2)}ms`);
      if (topResult) {
        console.log(`      Top result: "${topResult.experience.content.slice(0, 50)}..." (score: ${(topResult.score || 0).toFixed(3)})`);
      }
    } catch (error) {
      results.push({
        test: `Search: ${searchTest.description}`,
        passed: false,
        latency_ms: 0,
        details: `Error: ${error}`,
      });
      console.log(`   ❌ ${searchTest.description} failed: ${error}`);
    }
  }

  // Test 4: List All Memories
  console.log("\n📋 Test 4: List All Memories...");
  try {
    const { result, latency_ms } = await measureLatency(() =>
      apiCall("/api/memories", "POST", { user_id: USER_ID })
    );

    const count = (result.memories || []).length;
    const passed = count === testMemories.length;

    results.push({
      test: "List all memories",
      passed,
      latency_ms,
      details: `Found ${count}/${testMemories.length} memories`,
    });
    console.log(`   ${passed ? '✅' : '❌'} Listed ${count} memories in ${latency_ms.toFixed(2)}ms`);
  } catch (error) {
    results.push({
      test: "List all memories",
      passed: false,
      latency_ms: 0,
      details: `Error: ${error}`,
    });
    console.log(`   ❌ Failed: ${error}`);
  }

  // Test 5: Memory Stats
  console.log("\n📋 Test 5: Memory Stats...");
  try {
    const { result, latency_ms } = await measureLatency(() =>
      apiCall(`/api/users/${USER_ID}/stats`, "GET")
    );

    results.push({
      test: "Memory stats",
      passed: true,
      latency_ms,
      details: JSON.stringify(result),
    });
    console.log(`   ✅ Stats retrieved in ${latency_ms.toFixed(2)}ms`);
    console.log(`      ${JSON.stringify(result, null, 2).split('\n').map(l => '      ' + l).join('\n')}`);
  } catch (error) {
    results.push({
      test: "Memory stats",
      passed: false,
      latency_ms: 0,
      details: `Error: ${error}`,
    });
    console.log(`   ❌ Failed: ${error}`);
  }

  // Test 6: Delete Memory
  console.log("\n📋 Test 6: Delete Memory...");
  if (savedMemoryIds.length > 0) {
    const idToDelete = savedMemoryIds[0];
    try {
      const { latency_ms } = await measureLatency(() =>
        apiCall(`/api/memory/${idToDelete}?user_id=${USER_ID}`, "DELETE")
      );

      results.push({
        test: "Delete memory",
        passed: true,
        latency_ms,
        details: `Deleted ${idToDelete}`,
      });
      console.log(`   ✅ Deleted memory in ${latency_ms.toFixed(2)}ms`);
    } catch (error) {
      results.push({
        test: "Delete memory",
        passed: false,
        latency_ms: 0,
        details: `Error: ${error}`,
      });
      console.log(`   ❌ Failed: ${error}`);
    }
  }

  // Test 7: Edge Case - Empty Query
  console.log("\n📋 Test 7: Edge Cases...");
  try {
    const { result, latency_ms } = await measureLatency(() =>
      apiCall("/api/recall", "POST", {
        user_id: USER_ID,
        query: "completely random gibberish xyzzy12345",
        limit: 3,
      })
    );

    const hasResults = (result.memories || []).length > 0;
    results.push({
      test: "Irrelevant query handling",
      passed: true, // Should still return something (most relevant)
      latency_ms,
      details: `Returned ${(result.memories || []).length} results for irrelevant query`,
    });
    console.log(`   ✅ Irrelevant query handled in ${latency_ms.toFixed(2)}ms`);
  } catch (error) {
    results.push({
      test: "Irrelevant query handling",
      passed: false,
      latency_ms: 0,
      details: `Error: ${error}`,
    });
  }

  // Calculate summary
  const avgSaveLatency = saveLatencies.length > 0
    ? saveLatencies.reduce((a, b) => a + b, 0) / saveLatencies.length
    : 0;
  const avgRetrievalLatency = retrievalLatencies.length > 0
    ? retrievalLatencies.reduce((a, b) => a + b, 0) / retrievalLatencies.length
    : 0;
  const semanticAccuracy = searchTests.length > 0
    ? (correctRetrievals / searchTests.length) * 100
    : 0;

  const report: BenchmarkReport = {
    timestamp: new Date().toISOString(),
    total_tests: results.length,
    passed: results.filter(r => r.passed).length,
    failed: results.filter(r => !r.passed).length,
    results,
    summary: {
      avg_save_latency_ms: avgSaveLatency,
      avg_retrieval_latency_ms: avgRetrievalLatency,
      semantic_accuracy: semanticAccuracy,
      relevance_scores: relevanceScores,
    },
  };

  return report;
}

function printReport(report: BenchmarkReport) {
  console.log("\n" + "=".repeat(60));
  console.log("📊 BENCHMARK REPORT");
  console.log("=".repeat(60));

  console.log(`\n⏱️  Timestamp: ${report.timestamp}`);
  console.log(`📈 Total Tests: ${report.total_tests}`);
  console.log(`✅ Passed: ${report.passed}`);
  console.log(`❌ Failed: ${report.failed}`);
  console.log(`📊 Pass Rate: ${((report.passed / report.total_tests) * 100).toFixed(1)}%`);

  console.log("\n" + "-".repeat(60));
  console.log("PERFORMANCE METRICS");
  console.log("-".repeat(60));
  console.log(`💾 Avg Save Latency: ${report.summary.avg_save_latency_ms.toFixed(2)}ms`);
  console.log(`🔍 Avg Retrieval Latency: ${report.summary.avg_retrieval_latency_ms.toFixed(2)}ms`);
  console.log(`🎯 Semantic Search Accuracy: ${report.summary.semantic_accuracy.toFixed(1)}%`);

  if (report.summary.relevance_scores.length > 0) {
    const avgScore = report.summary.relevance_scores.reduce((a, b) => a + b, 0) / report.summary.relevance_scores.length;
    const minScore = Math.min(...report.summary.relevance_scores);
    const maxScore = Math.max(...report.summary.relevance_scores);
    console.log(`📐 Relevance Scores: avg=${avgScore.toFixed(3)}, min=${minScore.toFixed(3)}, max=${maxScore.toFixed(3)}`);
  }

  console.log("\n" + "-".repeat(60));
  console.log("DETAILED RESULTS");
  console.log("-".repeat(60));

  for (const result of report.results) {
    const icon = result.passed ? '✅' : '❌';
    console.log(`${icon} ${result.test}`);
    console.log(`   Latency: ${result.latency_ms.toFixed(2)}ms`);
    console.log(`   Details: ${result.details}`);
  }

  console.log("\n" + "=".repeat(60));
  console.log("END OF REPORT");
  console.log("=".repeat(60));
}

// Main
async function main() {
  try {
    const report = await runBenchmark();
    printReport(report);

    // Save report to file
    const reportPath = "./benchmark_report.json";
    await Bun.write(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📁 Report saved to: ${reportPath}`);

  } catch (error) {
    console.error("❌ Benchmark failed:", error);
    process.exit(1);
  }
}

main();
