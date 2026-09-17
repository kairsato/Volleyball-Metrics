// Records short GIFs of the Setup -> Court Calibration / Player
// Identification / Scoring Determination flows for docs/assets/.
//
// SAFETY: these jobs are real user data. These flows demonstrate the
// interaction (dragging a calibration point, opening the name picker,
// clicking timeline cells) but deliberately never click a final "Set Court
// Identification" / "Confirm Player Identification" / "Confirm Scoring"
// button. Selecting a scoring Method and running Heuristic/CV analysis
// does write scoreConfig + a draft (unconfirmed) scoreResult to the
// backend - see ScoreSection.tsx's saveConfig/handleAnalyze - but that is
// not a final confirmation (scoreConfig.confirmed stays false, "Redo
// Scoring" is what's gated behind confirmation, not the draft result) and
// is unavoidable to make the ScoreTrackEditor timeline exist at all for
// this never-before-scored job, per direct inspection of ScoreSection.tsx.
//
// Reference job per flow: player-id was moved to b44b4bd3ab2d (its
// player_config.json is unconfirmed, with 122 still-unidentified
// tracklets, so the flow works unchanged). court and scoring stayed on
// 09735a8b2de0 deliberately - on b44b4bd3ab2d both court.json and
// score_config.json are already confirmed:true, which means
// CalibrationPanel.tsx / ScoreSection.tsx render a LockOverlay over the
// handles / Method dropdown / ScoreTrackEditor (see LockOverlay.tsx) that
// intercepts every click and opens a "Redo ...?" dialog instead. Actually
// confirming that dialog (handleRedoConfirm / the scoring equivalent)
// calls the backend to flip confirmed back to false - a real mutation of
// confirmed production data - so there is no way to demo dragging a
// handle or clicking timeline cells on that job without either doing
// nothing (overlay swallows the interaction) or risking that mutation.
//
// Usage (from Frontend/):
//   node scripts/record-calibration-demos.mjs [court|player-id|scoring|all]

import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, "..", "..", "docs", "assets");
const TMP_ROOT = path.join(os.tmpdir(), "vva-calibration-demos");

const BASE = process.env.DEMO_BASE_URL ?? "http://127.0.0.1:3000";
// Per-flow reference job - see the safety comment above for why court and
// scoring stayed on the old job while player-id moved to the new one.
const JOB_IDS = {
  court: "09735a8b2de0",
  "player-id": "b44b4bd3ab2d",
  scoring: "09735a8b2de0",
};

const VIEWPORT = { width: 1100, height: 780 };
const GIF_FPS = 10;
const GIF_WIDTH = 560;

// No redaction anywhere - every demo GIF (this script and
// record-frontend-demos.mjs) shows real footage/photos/names as-is.

async function waitSettled(page, { timeout = 60000, settleMs = 2000 } = {}) {
  await page.waitForFunction(() => document.querySelectorAll(".MuiSkeleton-root").length === 0, null, { timeout });
  await page.waitForLoadState("networkidle", { timeout }).catch(() => {});
  await page.waitForTimeout(settleMs);
}

async function recordFlow(browser, name, run) {
  const videoDir = path.join(TMP_ROOT, name);
  await fs.rm(videoDir, { recursive: true, force: true });
  await fs.mkdir(videoDir, { recursive: true });

  const context = await browser.newContext({ viewport: VIEWPORT, recordVideo: { dir: videoDir, size: VIEWPORT } });
  const page = await context.newPage();
  page.on("dialog", (d) => d.dismiss().catch(() => {}));
  try {
    await run(page);
  } finally {
    await context.close();
  }

  const files = await fs.readdir(videoDir);
  const webm = files.find((f) => f.endsWith(".webm"));
  if (!webm) throw new Error(`recordFlow(${name}): no .webm produced`);
  return path.join(videoDir, webm);
}

function convertToGif(webmPath, gifPath) {
  const filters = `fps=${GIF_FPS},scale=${GIF_WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse`;
  execFileSync("ffmpeg", ["-y", "-i", webmPath, "-vf", filters, "-loop", "0", gifPath], { stdio: "inherit" });
}

async function sizeOf(filePath) {
  return (await fs.stat(filePath)).size;
}

const FLOWS = {
  court: {
    gifName: "frontend-court-calibration.gif",
    async run(page) {
      await page.goto(`${BASE}/game/setup?job=${JOB_IDS.court}`, { waitUntil: "domcontentloaded" });
      await waitSettled(page, { settleMs: 1500 });
      await page.getByText("Court Calibration", { exact: true }).click();
      await waitSettled(page, { settleMs: 1500 });

      // This job's calibration is currently unlocked/editable (button reads
      // "Set Court Identification", not "Redo") - confirmed via a live DOM
      // probe - so no unlock step is needed here. Drag the "Close Line
      // Left" point handle (POINT_TITLES index 2 - see CalibrationPanel.tsx)
      // a short distance to demonstrate the interaction and the magnifier
      // loupe, without ever clicking the Set/Redo button.
      const handle = page.locator(".calibration-handle").nth(2);
      const box = await handle.boundingBox();
      if (box) {
        const startX = box.x + box.width / 2;
        const startY = box.y + box.height / 2;
        await page.mouse.move(startX, startY);
        await page.mouse.down();
        await page.mouse.move(startX + 30, startY - 20, { steps: 12 });
        await page.waitForTimeout(600);
        await page.mouse.move(startX + 55, startY - 35, { steps: 12 });
        await page.waitForTimeout(800);
        await page.mouse.up();
      }
      await page.waitForTimeout(1200);
    },
  },

  "player-id": {
    gifName: "frontend-player-id.gif",
    async run(page) {
      await page.goto(`${BASE}/game/setup?job=${JOB_IDS["player-id"]}`, { waitUntil: "domcontentloaded" });
      await waitSettled(page, { settleMs: 1500 });
      // The Setup landing card for this section may render locked/disabled
      // (Player Identification requires court calibration to be confirmed
      // first - see SetupTab.tsx) even though the sub-page itself loads
      // fine via direct navigation, so go there directly rather than
      // relying on the card being clickable.
      await page.goto(`${BASE}/game/setup/player-identification?job=${JOB_IDS["player-id"]}`, {
        waitUntil: "domcontentloaded",
      });
      await waitSettled(page, { settleMs: 1500, timeout: 60000 });

      // A "Is the court definition accurate?" dialog gates this page behind
      // a one-time confirmation (see PlayerIdentificationPage.tsx) - dismiss
      // it via "Looks right, continue" so the player cards underneath
      // become clickable, without ever touching "Review calibration".
      const accuracyDialog = page.getByText("Looks right, continue");
      if ((await accuracyDialog.count()) > 0) {
        await accuracyDialog.click();
        await page.waitForTimeout(500);
      }

      // 122 unidentified players on this job - "Confirm Player
      // Identification" is disabled regardless, so there is no risk of an
      // accidental confirm here. Each tile is a Card with onClick=onSelect
      // (see UnidentifiedPlayersSection.tsx) - click the first two tiles
      // under "Unidentified players" to select them, open the name picker
      // to show it populated, then close without assigning.
      const cards = page.locator(".MuiCard-root");
      const count = await cards.count();
      let clicked = 0;
      for (let i = 0; i < count && clicked < 2; i++) {
        const el = cards.nth(i);
        const box = await el.boundingBox();
        if (box && box.y > 150) {
          await el.click();
          clicked++;
          await page.waitForTimeout(400);
        }
      }
      console.log(`[player-id] selected ${clicked} tiles`);

      const nameField = page.getByPlaceholder("Pick a name");
      if (clicked > 0 && (await nameField.count()) > 0 && (await nameField.isEnabled())) {
        await nameField.click();
        await page.waitForTimeout(700);
        await page.keyboard.press("Escape");
      }
      await page.waitForTimeout(1000);
    },
  },

  scoring: {
    gifName: "frontend-scoring.gif",
    async run(page) {
      await page.goto(`${BASE}/game/setup?job=${JOB_IDS.scoring}`, { waitUntil: "domcontentloaded" });
      await waitSettled(page, { settleMs: 1500 });
      await page.goto(`${BASE}/game/setup/scoring-determination?job=${JOB_IDS.scoring}`, {
        waitUntil: "domcontentloaded",
      });
      await waitSettled(page, { settleMs: 1500 });

      // This job has never been scored, and ScoreTrackEditor only renders
      // once a scoreResult exists (see ScoreSection.tsx) - the "Analyze"
      // button that produces one is only shown for Heuristic/Computer
      // Vision, not Manual. Use Heuristic first (fast, ball-trajectory
      // based, no scoreboard OCR model) purely to populate the timeline,
      // then switch the dropdown to Computer Vision per the requested
      // demo, without ever clicking Confirm/Redo Scoring.
      await page.getByLabel("Method").click();
      await page.waitForTimeout(400);
      await page.getByRole("option", { name: /^Heuristic/i }).click();
      await page.waitForTimeout(800);

      await page.getByLabel("Team 1").click();
      await page.waitForTimeout(400);
      await page.getByRole("option").first().click();
      await page.waitForTimeout(600);

      const analyzeBtn = page.getByRole("button", { name: "Analyze" });
      await analyzeBtn.click();
      // Poll (see ScoreSection.startPolling, 2s interval) until a real
      // result renders or we give up - heuristic scoring over 51 rallies
      // on this dev machine has been observed finishing well under a
      // minute.
      await page
        .locator("text=/^Set 1|^Match 1/i")
        .first()
        .waitFor({ state: "visible", timeout: 90000 })
        .catch(() => {});
      await page.waitForTimeout(1500);

      // Now switch the Method dropdown to Computer Vision, per the
      // requested demo - this auto-opens the scoreboard region dialog;
      // close it without drawing rather than fight canvas coordinates.
      await page.getByLabel("Method").click();
      await page.waitForTimeout(400);
      await page.getByRole("option", { name: /Computer Vision/i }).click();
      await page.waitForTimeout(1200);
      const dialog = page.locator('div[role="dialog"]');
      if ((await dialog.count()) > 0) {
        const cancelBtn = dialog.getByRole("button", { name: "Cancel" });
        if ((await cancelBtn.count()) > 0) await cancelBtn.click();
        await page.waitForTimeout(600);
      }

      // Scroll to the timeline and click a couple of rally cells to cycle
      // each team's winner marker - cycleTeam1/cycleTeam2 in
      // ScoreTrackEditor.tsx, a manual override independent of method.
      await page.mouse.wheel(0, 500);
      await page.waitForTimeout(600);
      const trackCells = page.locator('[data-rally-index], [role="button"]').filter({ hasText: "" });
      const clickable = page.locator("canvas, svg").last();
      const box = await clickable.boundingBox();
      if (box) {
        for (const dx of [40, 70, 100]) {
          await page.mouse.click(box.x + dx, box.y + box.height * 0.55);
          await page.waitForTimeout(400);
        }
      }
      await page.waitForTimeout(1200);
    },
  },
};

async function main() {
  const requested = process.argv.slice(2);
  const names = requested.length > 0 && requested[0] !== "all" ? requested : Object.keys(FLOWS);

  for (const name of names) {
    if (!FLOWS[name]) {
      console.error(`Unknown flow "${name}" - choices: ${Object.keys(FLOWS).join(", ")}, all`);
      process.exitCode = 1;
      return;
    }
  }

  await fs.mkdir(OUT_DIR, { recursive: true });
  const browser = await chromium.launch();

  try {
    for (const name of names) {
      const flow = FLOWS[name];
      console.log(`\n=== ${name} ===`);
      const webmPath = await recordFlow(browser, name, flow.run);
      const gifPath = path.join(OUT_DIR, flow.gifName);
      convertToGif(webmPath, gifPath);
      const bytes = await sizeOf(gifPath);
      console.log(`Wrote ${gifPath} (${(bytes / 1024).toFixed(0)} KB)`);
      await fs.rm(path.join(TMP_ROOT, name), { recursive: true, force: true });
    }
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
