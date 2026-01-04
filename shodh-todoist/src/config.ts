export const config = {
  todoist: {
    apiToken: process.env.TODOIST_API_TOKEN || "",
    apiUrl: "https://api.todoist.com/rest/v2",
  },
  shodh: {
    apiUrl: process.env.SHODH_API_URL || "http://127.0.0.1:3030",
    apiKey: process.env.SHODH_API_KEY || "sk-shodh-dev-local-testing-key",
    userId: process.env.SHODH_USER_ID || "claude-code",
  },
  sync: {
    intervalMs: parseInt(process.env.SYNC_INTERVAL_MS || "30000"),
  },
};
