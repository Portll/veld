import { describe, expect, it } from "vitest";
import {
  autoStreamText,
  buildProgressBar,
  extractStringContextFromArgs,
  formatResetTokenSessionText,
  formatTokenStatusText,
  normalizeLimit,
  shouldAppendProactiveContext,
} from "../index-helpers";

describe("buildProgressBar", () => {
  it("builds 20-char bar", () => {
    const bar = buildProgressBar(50);
    expect(bar.length).toBe(20);
    expect(bar).toContain("█");
    expect(bar).toContain("░");
  });

  it("clamps percent bounds", () => {
    expect(buildProgressBar(-10)).toBe("░".repeat(20));
    expect(buildProgressBar(150)).toBe("█".repeat(20));
  });
});

describe("formatTokenStatusText", () => {
  it("formats healthy status", () => {
    const text = formatTokenStatusText(
      { tokens: 100, budget: 1000, percent: 0.1, alert: null },
      5,
    );
    expect(text).toContain("Token Status");
    expect(text).toContain("Used: 100 tokens");
    expect(text).toContain("Budget: 1,000 tokens");
    expect(text).toContain("Remaining: 900 tokens");
    expect(text).toContain("Context window healthy");
  });

  it("formats alert status", () => {
    const text = formatTokenStatusText(
      { tokens: 950, budget: 1000, percent: 0.95, alert: "context_90_percent" },
      12,
    );
    expect(text).toContain("ALERT");
    expect(text).toContain("95% used");
  });
});

describe("normalizeLimit", () => {
  it("defaults when undefined", () => {
    expect(normalizeLimit(undefined)).toBe(10);
  });

  it("enforces minimum", () => {
    expect(normalizeLimit(0)).toBe(1);
    expect(normalizeLimit(-5)).toBe(1);
  });

  it("enforces maximum", () => {
    expect(normalizeLimit(999, 250)).toBe(250);
  });

  it("floors decimals", () => {
    expect(normalizeLimit(7.9)).toBe(7);
  });
});

describe("formatResetTokenSessionText", () => {
  it("formats reset text with prior usage and budget", () => {
    const text = formatResetTokenSessionText(12500, 100000);
    expect(text).toContain("Token Session Reset");
    expect(text).toContain("Previous: 12,500 tokens");
    expect(text).toContain("Current: 0 tokens");
    expect(text).toContain("Budget: 100,000 tokens");
    expect(text).toContain("Counter cleared");
  });
});

describe("shouldAppendProactiveContext", () => {
  it("returns false for excluded memory-management tools", () => {
    expect(shouldAppendProactiveContext("remember")).toBe(false);
    expect(shouldAppendProactiveContext("recall")).toBe(false);
    expect(shouldAppendProactiveContext("memory_stats")).toBe(false);
  });

  it("returns true for normal tools", () => {
    expect(shouldAppendProactiveContext("search_code")).toBe(true);
    expect(shouldAppendProactiveContext("read_file")).toBe(true);
  });
});

describe("extractStringContextFromArgs", () => {
  it("extracts only sufficiently long string values", () => {
    const context = extractStringContextFromArgs({
      short: "abc",
      good: "this should be included",
      count: 42,
      nested: { x: 1 },
      alsoGood: "another long value",
    });
    expect(context).toContain("this should be included");
    expect(context).toContain("another long value");
    expect(context).not.toContain("abc");
  });

  it("caps output length", () => {
    const veryLong = "x".repeat(3000);
    const context = extractStringContextFromArgs({ a: veryLong }, 1, 1000);
    expect(context.length).toBe(1000);
  });

  it("returns empty string for non-object args", () => {
    expect(extractStringContextFromArgs(null)).toBe("");
    expect(extractStringContextFromArgs("text")).toBe("");
  });
});

describe("autoStreamText", () => {
  const question = "why was the launchd restart changed to kickstart instead of relaunch";

  it("never streams a query, whatever the tool", () => {
    for (const tool of ["recall", "quick_recall", "query", "topic", "what_i_know", "list_todos"]) {
      expect(autoStreamText(tool, { query: question })).toBeNull();
    }
  });

  it("prefers nothing over a query even when content is also present", () => {
    expect(autoStreamText("list_todos", { query: question, context: question })).toBeNull();
  });

  it("streams content and context from tools that carry conversation", () => {
    expect(autoStreamText("add_todo", { content: "Pin the commit-msg gate to a committed revision" })).toBe(
      "Pin the commit-msg gate to a committed revision",
    );
    expect(autoStreamText("set_reminder", { context: "Re-run the recall gate after the merge lands" })).toBe(
      "Re-run the recall gate after the merge lands",
    );
  });

  it("does not stream remember, which stores its own content", () => {
    expect(autoStreamText("remember", { content: "a decision worth keeping for later sessions" })).toBeNull();
  });

  it("skips the meta tools and short or missing text", () => {
    expect(autoStreamText("token_status", { context: "a long enough piece of context text" })).toBeNull();
    expect(autoStreamText("add_todo", { content: "too short" })).toBeNull();
    expect(autoStreamText("add_todo", {})).toBeNull();
    expect(autoStreamText("add_todo", undefined)).toBeNull();
    expect(autoStreamText("add_todo", { content: 42 })).toBeNull();
  });
});
