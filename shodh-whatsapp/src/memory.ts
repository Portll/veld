import { config } from "./config";

interface Memory {
  id: string;
  content: string;
  memory_type: string;
  relevance_score?: number;
  created_at: string;
}

interface ProactiveContextResponse {
  memories: Memory[];
  total_found: number;
}

interface RecallResponse {
  memories: Memory[];
}

export async function getProactiveContext(
  userId: string,
  context: string
): Promise<string> {
  try {
    console.log(`🔍 Fetching memory for user: ${userId}`);

    const response = await fetch(
      `${config.shodh.apiUrl}/api/proactive_context`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": config.shodh.apiKey,
        },
        body: JSON.stringify({
          user_id: userId,
          context,
          auto_ingest: true,
          max_results: 10,
          semantic_threshold: 0.4,
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Memory API error: ${response.status} - ${errorText}`);
      return "";
    }

    const data = (await response.json()) as ProactiveContextResponse;
    console.log(`🔍 Memory API returned ${data.memories?.length || 0} memories (total: ${data.total_found || 0})`);

    if (!data.memories || data.memories.length === 0) {
      return "";
    }

    return data.memories
      .map((m) => `[${m.memory_type}] ${m.content}`)
      .join("\n");
  } catch (error) {
    console.error("Failed to get proactive context:", error);
    return "";
  }
}

export async function remember(
  userId: string,
  content: string,
  memoryType: string = "Conversation",
  tags: string[] = []
): Promise<boolean> {
  try {
    const response = await fetch(`${config.shodh.apiUrl}/api/remember`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.shodh.apiKey,
      },
      body: JSON.stringify({
        user_id: userId,
        content,
        memory_type: memoryType,
        tags,
        source_type: "api",
      }),
    });

    return response.ok;
  } catch (error) {
    console.error("Failed to store memory:", error);
    return false;
  }
}

export async function recall(userId: string, query: string): Promise<string> {
  try {
    const response = await fetch(`${config.shodh.apiUrl}/api/recall`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.shodh.apiKey,
      },
      body: JSON.stringify({
        user_id: userId,
        query,
        limit: 5,
        mode: "hybrid",
      }),
    });

    if (!response.ok) {
      return "";
    }

    const data = (await response.json()) as RecallResponse;

    if (!data.memories || data.memories.length === 0) {
      return "";
    }

    return data.memories.map((m) => m.content).join("\n");
  } catch (error) {
    console.error("Failed to recall:", error);
    return "";
  }
}

export async function getContextSummary(userId: string): Promise<string> {
  try {
    const response = await fetch(`${config.shodh.apiUrl}/api/context_summary`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.shodh.apiKey,
      },
      body: JSON.stringify({
        user_id: userId,
        include_learnings: true,
        include_decisions: true,
        include_context: true,
        max_items: 5,
      }),
    });

    if (!response.ok) {
      return "";
    }

    const data = await response.json();
    const parts: string[] = [];

    if (data.learnings?.length > 0) {
      parts.push("Recent learnings: " + data.learnings.join("; "));
    }
    if (data.decisions?.length > 0) {
      parts.push("Recent decisions: " + data.decisions.join("; "));
    }
    if (data.context?.length > 0) {
      parts.push("Context: " + data.context.join("; "));
    }

    return parts.join("\n");
  } catch (error) {
    console.error("Failed to get context summary:", error);
    return "";
  }
}
