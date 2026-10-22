import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// The TypeScript server and `veld serve` (src/mcp/) expose one tool set. This reads both from
// source so the check runs under vitest without a backend; tests/mcp_tool_parity.rs is the
// same check from the Rust side, where the live router is read as well.

const mcpServerDir = resolve(__dirname, "..");
const repoRoot = resolve(mcpServerDir, "..");

function typescriptToolNames(): Set<string> {
  const source = readFileSync(resolve(mcpServerDir, "index.ts"), "utf8");
  const start = source.indexOf("setRequestHandler(ListToolsRequestSchema");
  expect(start).toBeGreaterThanOrEqual(0);
  const region = source.slice(start);
  const next = region.indexOf("setRequestHandler(", 1);
  const listing = next === -1 ? region : region.slice(0, next);
  const names = new Set<string>();
  for (const line of listing.split("\n")) {
    const match = /^\s*name: "([a-z_][a-z0-9_]*)",$/.exec(line);
    if (match) names.add(match[1]);
  }
  return names;
}

function rustToolNames(): Set<string> {
  const names = new Set<string>();
  for (const file of ["tools.rs", "tools_memory.rs", "tools_work.rs"]) {
    const source = readFileSync(resolve(repoRoot, "src/mcp", file), "utf8");
    const pattern = /#\[tool\([\s\S]*?\bfn\s+([a-z_][a-z0-9_]*)\s*\(/g;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(source)) !== null) names.add(match[1]);
  }
  return names;
}

describe("MCP tool parity", () => {
  it("the TypeScript server and veld serve declare the same tools", () => {
    const typescript = typescriptToolNames();
    const rust = rustToolNames();
    const onlyTypescript = [...typescript].filter((n) => !rust.has(n)).sort();
    const onlyRust = [...rust].filter((n) => !typescript.has(n)).sort();
    expect({ onlyTypescript, onlyRust }).toEqual({ onlyTypescript: [], onlyRust: [] });
    expect(typescript.size).toBeGreaterThanOrEqual(60);
  });
});
