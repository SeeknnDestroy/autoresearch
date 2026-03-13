import { expect, test } from "@playwright/test";

test("Autoresearch Studio runs baseline and candidate flows", async ({ page }, testInfo) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: /Start simple/i })).toBeVisible();
  await expect(page.getByRole("button", { name: "Play Mode" })).toHaveClass(/is-active/);
  await expect(page.getByRole("button", { name: "Run a tiny experiment" })).toBeVisible();
  await expect(page.locator("#play-steps .step-card")).toHaveCount(4);
  await expect(page.locator("#play-repo-map .repo-card")).toHaveCount(3);

  await page.getByRole("button", { name: "Run a tiny experiment" }).click();

  await expect(page.locator("#play-score-value")).not.toHaveText("--", { timeout: 20_000 });
  await expect(page.locator("#play-verdict-title")).not.toContainText("Nothing has happened yet", { timeout: 20_000 });
  await expect(page.locator("#play-summary-runs")).toContainText("experiment", { timeout: 20_000 });

  await page.screenshot({ path: testInfo.outputPath("studio-desktop.png"), fullPage: true });

  await page.getByRole("button", { name: "Advanced Lab" }).click();
  await expect(page.getByRole("button", { name: "Advanced Lab" })).toHaveClass(/is-active/);
  await expect(page.locator("#stage-headline")).toBeVisible();
  await expect(page.locator(".run-card").first()).toBeVisible();
  await expect(page.locator("#detail-title")).not.toHaveText("No run selected");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Play Mode" }).click();
  await expect(page.locator("#play-score-value")).toBeVisible();
  await expect(page.locator("#play-summary-runs")).toBeVisible();

  await page.screenshot({ path: testInfo.outputPath("studio-mobile.png"), fullPage: true });
});
