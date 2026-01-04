export const config = {
  telegram: {
    token: process.env.TELEGRAM_BOT_TOKEN || "",
    ownerId: parseInt(process.env.OWNER_ID || "0"),
  },
  security: {
    pin: process.env.SECURITY_PIN || "",
  },
  shodh: {
    apiUrl: process.env.SHODH_API_URL || "http://127.0.0.1:3030",
    apiKey: process.env.SHODH_API_KEY || "sk-shodh-dev-local-testing-key",
  },
  debug: process.env.DEBUG === "true",
};

export function isOwner(userId: number): boolean {
  return userId === config.telegram.ownerId;
}

export function verifyPin(pin: string): boolean {
  if (!config.security.pin) return true;
  return pin === config.security.pin;
}
