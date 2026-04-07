import { describe, expect, it } from "bun:test";
import {
  buildPreToolContext,
  formatRelativeTime,
  formatMemoriesForContext,
  isErrorOutput,
} from "./memory-hook";

describe("formatRelativeTime", () => {
  it("returns today for current date", () => {
    const nowIso = new Date().toISOString();
    expect(formatRelativeTime(nowIso)).toBe("today");
  });

  it("returns yesterday for one day old date", () => {
    const d = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(d)).toBe("yesterday");
  });

  it("returns Xd ago for dates under a week", () => {
    const d = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(d)).toBe("3d ago");
  });

  it("returns calendar date for older memories", () => {
    const d = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString();
    const value = formatRelativeTime(d);
    expect(value).not.toContain("ago");
    expect(value).not.toBe("today");
    expect(value).not.toBe("yesterday");
  });
});

describe("formatMemoriesForContext", () => {
  it("returns empty string for empty input", () => {
    expect(formatMemoriesForContext([])).toBe("");
  });

  it("formats score, relative time, and content", () => {
    const memories = [
      {
        id: "m1",
        content: "Remember to review deployment checklist before release",
        memory_type: "Task",
        score: 0.83,
        importance: 0.7,
        created_at: new Date().toISOString(),
        tags: ["deploy"],
        relevance_reason: "matches query",
        matched_entities: ["release"],
      },
    ];

    const out = formatMemoriesForContext(memories);
    expect(out).toContain("[83%]");
    expect(out).toContain("(today)");
    expect(out).toContain("Remember to review deployment checklist");
  });

  it("truncates long memory content at 120 chars with ellipsis", () => {
    const longContent = "x".repeat(130);
    const memories = [
      {
        id: "m2",
        content: longContent,
        memory_type: "Observation",
        score: 0.5,
        importance: 0.5,
        created_at: new Date().toISOString(),
        tags: [],
        relevance_reason: "test",
        matched_entities: [],
      },
    ];

    const out = formatMemoriesForContext(memories);
    expect(out).toContain("...");
    expect(out.length).toBeGreaterThan(120);
  });

  it("formats multiple memories as multiple bullet lines", () => {
    const now = new Date().toISOString();
    const out = formatMemoriesForContext([
      {
        id: "m1",
        content: "A",
        memory_type: "Observation",
        score: 0.2,
        importance: 0.1,
        created_at: now,
        tags: [],
        relevance_reason: "x",
        matched_entities: [],
      },
      {
        id: "m2",
        content: "B",
        memory_type: "Observation",
        score: 0.3,
        importance: 0.1,
        created_at: now,
        tags: [],
        relevance_reason: "x",
        matched_entities: [],
      },
    ]);
    expect((out.match(/•/g) || []).length).toBe(2);
  });
});

describe("buildPreToolContext", () => {
  it("builds edit context", () => {
    expect(buildPreToolContext("Edit", { file_path: "src/main.ts" })).toBe("Editing file: src/main.ts");
  });

  it("builds bash context and truncates command", () => {
    const cmd = "x".repeat(140);
    const value = buildPreToolContext("Bash", { command: cmd });
    expect(value.startsWith("Running command: ")).toBe(true);
    expect(value.length).toBeLessThanOrEqual("Running command: ".length + 100);
  });

  it("falls back to generic context", () => {
    expect(buildPreToolContext("Read", {})).toBe("About to use Read");
  });
});

describe("isErrorOutput", () => {
  it("detects error/failure strings", () => {
    expect(isErrorOutput("fatal error occurred")).toBe(true);
    expect(isErrorOutput("Operation FAILED quickly")).toBe(true);
    expect(isErrorOutput("This failed with code 1")).toBe(true);
  });

  it("returns false on clean output", () => {
    expect(isErrorOutput("Command completed successfully")).toBe(false);
  });
});
