import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 780 } });
await page.goto("http://127.0.0.1:5173/game/setup/player-identification?job=09735a8b2de0", { waitUntil: "domcontentloaded" });
await page.waitForTimeout(4000);
await page.screenshot({ path: "player-id-debug.png", fullPage: false });
const dialogText = await page.locator('div[role="dialog"]').allTextContents().catch(() => []);
console.log("DIALOG TEXT:", JSON.stringify(dialogText));
await browser.close();
