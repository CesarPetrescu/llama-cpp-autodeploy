import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../../..");
const outputDir = path.join(repoRoot, "docs", "screenshots");

const baseUrl = process.env.WEB_BASE_URL || "http://127.0.0.1:8787";
const token = process.env.WEB_BEARER_TOKEN || "";
const apiBase = process.env.WEB_API_BASE ?? "";

async function openApp(page) {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "Overview", exact: true }).waitFor({
    state: "visible",
    timeout: 15000,
  });
}

async function navigateViaUi(page, routeLabel) {
  if (routeLabel === "Overview") {
    return;
  }
  await page.getByRole("link", { name: new RegExp(routeLabel, "i") }).first().click();
  await page.getByRole("heading", { name: routeLabel, exact: true }).waitFor({
    state: "visible",
    timeout: 15000,
  });
}

async function captureViewport(page, routeLabel, outputName) {
  await openApp(page);
  await navigateViaUi(page, routeLabel);
  await page.waitForTimeout(800);
  await page.screenshot({
    path: path.join(outputDir, outputName),
    animations: "disabled",
  });
}

async function captureGpuPanel(page, outputName) {
  await openApp(page);
  const gpuPanel = page
    .getByText("GPU pressure", { exact: true })
    .locator("xpath=ancestor::section[1]");
  await gpuPanel.scrollIntoViewIfNeeded();
  const firstSummary = gpuPanel.locator("details summary").first();
  if (await firstSummary.count()) {
    await firstSummary.click();
    await page.waitForTimeout(400);
  }
  await gpuPanel.screenshot({
    path: path.join(outputDir, outputName),
    animations: "disabled",
  });
}

async function main() {
  await fs.mkdir(outputDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1120 },
    deviceScaleFactor: 1,
  });

  await context.addInitScript(
    ({ currentToken, currentApiBase }) => {
      if (currentToken) {
        window.localStorage.setItem("llama_web_token", currentToken);
      }
      window.localStorage.setItem("llama_web_base", currentApiBase ?? "");
    },
    { currentToken: token, currentApiBase: apiBase },
  );

  const page = await context.newPage();

  try {
    await captureViewport(page, "Overview", "web-dashboard-overview.png");
    await captureGpuPanel(page, "web-dashboard-gpu.png");
    await captureViewport(page, "Instances", "web-instances.png");
    await captureViewport(page, "Builds", "web-builds.png");
  } finally {
    await browser.close();
  }

  console.log(`Wrote screenshots to ${outputDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
