import { expect, test } from "@playwright/test";

test("Autoresearch Studio runs baseline and candidate flows", async ({ page }, testInfo) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Autoresearch Studio" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Start Local Session" })).toBeVisible();
  await expect(page.locator("#stage-headline")).toContainText(/first baseline|waiting for its first baseline/i);

  await page.getByRole("button", { name: "Start Local Session" }).click();

  await expect(page.locator(".run-card").first()).toBeVisible({ timeout: 20_000 });
  await expect(page.locator(".run-card").nth(1)).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("#session-status")).toContainText(/running|completed|stopping|stopped/);
  await expect(page.locator("#session-status")).toContainText("completed", { timeout: 20_000 });
  await expect(page.locator("#detail-title")).not.toHaveText("No run selected", { timeout: 20_000 });
  await expect(page.locator(".run-card").nth(2)).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("#score-strip .score-pill")).toHaveCount(4);
  await expect(page.locator("#mutation-atlas .atlas-card").first()).toBeVisible();
  await expect(page.locator("#detail-stage")).toContainText(/analyst|completed|runner|implementer/i);
  await expect(page.locator(".metric-value").first()).not.toHaveText("--");

  await page.screenshot({ path: testInfo.outputPath("studio-desktop.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("button", { name: "Start Local Session" })).toBeVisible();
  await expect(page.locator(".metrics-grid")).toBeVisible();
  await expect(page.locator("#mutation-card")).toBeVisible();

  await page.screenshot({ path: testInfo.outputPath("studio-mobile.png"), fullPage: true });
});
