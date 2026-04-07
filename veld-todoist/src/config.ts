export const config = {
  todoist: {
    apiToken: process.env.TODOIST_API_TOKEN || "",
    apiUrl: "https://api.todoist.com/rest/v2",
  },
  veld: {
    apiUrl: process.env.VELD_API_URL || "http://127.0.0.1:3030",
    apiKey: process.env.VELD_API_KEY || "sk-veld-dev-local-testing-key",
    userId: process.env.VELD_USER_ID || "claude-code",
  },
  sync: {
    intervalMs: parseInt(process.env.SYNC_INTERVAL_MS || "30000"),
  },
};
