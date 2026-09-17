import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("console", m => console.log("CONSOLE:", m.text()));
await page.goto("http://127.0.0.1:3000/game/setup?job=09735a8b2de0", { waitUntil: "domcontentloaded" });
await page.waitForTimeout(4000);
const body = await page.locator("body").innerText();
console.log("BODY TEXT:\n" + body);
await browser.close();
