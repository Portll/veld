import { defineConfig } from "vite";

// Tauri loads the built dist/ via the file://-style scheme. Keep the dev
// server pinned to a fixed port so tauri.conf.json can reference it.
export default defineConfig({
  clearScreen: false,
  server: {
    port: 1422,
    strictPort: true,
  },
  build: {
    target: "es2021",
    minify: "esbuild",
    sourcemap: false,
    emptyOutDir: true,
  },
});
