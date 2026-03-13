import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/ui",
  timeout: 45_000,
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    viewport: { width: 1600, height: 900 },
  },
  webServer: {
    command: "uv run python -m studio --host 127.0.0.1 --port 4173 --studio-dir .studio-playwright",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
    timeout: 45_000,
  },
});
