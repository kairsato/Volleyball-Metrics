// Records short GIFs of the frontend for docs/assets/, using Playwright to
// drive real Chromium against the already-running dev server (see
// docs/assets/tools/render_stage_demos.py for the same palette-based
// ffmpeg technique applied to job-replay clips instead of live UI).
//
// Usage (from Frontend/):
//   node scripts/record-frontend-demos.mjs [games|teams|players|game-results|all]
// Defaults to "all" when no flow name is given.
//
// Requires: the dev servers already running (backend :8000, frontend
// Vite :5180 with VITE_API_BASE pointed at the backend), and ffmpeg on
// PATH for the .webm -> .gif conversion.

import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, "..", "..", "docs", "assets");
const TMP_ROOT = path.join(os.tmpdir(), "vva-frontend-demos");

const BASE = process.env.DEMO_BASE_URL ?? "http://127.0.0.1:5180";
// The main cross-workstream reference job - large (100 rallies, 3 sets),
// fully processed, and has every rendition tier down to 480p (see
// transcode.py) so the game-results flow can demonstrate low-res preview
// playback below.
const RESULTS_JOB_ID = "399152820483";

const VIEWPORT = { width: 960, height: 600 };
// game-results records at a real 2K desktop size (2560x1440, i.e. QHD -
// what's commonly sold/called "2K") rather than the smaller viewport the
// other flows use, since this is the flow demonstrating the actual results
// page layout/video-preview experience, not just proving a list loads.
const RESULTS_VIEWPORT = { width: 2560, height: 1440 };
const GIF_FPS = 10;
const GIF_WIDTH = 480;
const RESULTS_GIF_WIDTH = 560;
const RESULTS_GIF_FPS = 8;

// --------------------------------------------------------------------------
// Shared waiting/scrolling helpers
// --------------------------------------------------------------------------

// Waits for any MUI Skeleton placeholders to be gone, in-flight requests to
// settle, and gives React a brief moment to paint the real content - used
// after every navigation/tab switch so a captured frame never lands on a
// loading state. Every page/tab in this app that has its own async fetch
// (GamesPage's job list, TeamsPage/PlayersPage's roster fetch, StatsTab/
// AnalyticsTab's per-tab fetch) renders a MUI <Skeleton> while waiting, so
// "no skeletons left in the DOM" is a reliable, page-agnostic loaded signal.
async function waitForLoaded(page, { settleMs = 1000, timeout = 30000 } = {}) {
  // Some tabs (Analytics in particular - see AnalyticsTab.tsx) fan out
  // several backend requests, and this dev machine's single-worker uvicorn
  // has been observed taking 60s+ under concurrent load from other work
  // happening in this repo right now - a silently-swallowed skeleton-wait
  // timeout here previously produced GIFs that captured the loading state
  // instead of the real one, so this now actually waits out the full
  // timeout rather than racing past it.
  await page.waitForFunction(() => document.querySelectorAll(".MuiSkeleton-root").length === 0, null, { timeout });
  await page.waitForLoadState("networkidle", { timeout }).catch(() => {});
  await page.waitForTimeout(settleMs);
}

// ResultsView's right-hand tab content column scrolls independently via its
// own `overflowY: auto` (see Frontend/src/components/ResultsView.tsx) - the
// page itself does not scroll. This tags that specific ancestor of the tab
// list so scrollContentDown() below scrolls the right element rather than
// window/document.body.
async function tagScrollContainer(page) {
  await page.evaluate(() => {
    const tablist = document.querySelector('[role="tablist"]');
    if (!tablist) return;
    let el = tablist.parentElement;
    while (el && getComputedStyle(el).overflowY !== "auto") {
      el = el.parentElement;
    }
    if (el) el.setAttribute("data-demo-scroll", "1");
  });
}

async function scrollContentDown(page) {
  const container = page.locator('[data-demo-scroll="1"]');
  if ((await container.count()) === 0) return;
  await container.evaluate((el) => el.scrollTo({ top: 0, behavior: "auto" }));
  await page.waitForTimeout(200);
  await container.evaluate((el) => el.scrollTo({ top: Math.min(420, el.scrollHeight), behavior: "smooth" }));
  await page.waitForTimeout(700);
}

// --------------------------------------------------------------------------
// Recording / conversion plumbing
// --------------------------------------------------------------------------

async function recordFlow(browser, name, run, { viewport = VIEWPORT } = {}) {
  const videoDir = path.join(TMP_ROOT, name);
  await fs.rm(videoDir, { recursive: true, force: true });
  await fs.mkdir(videoDir, { recursive: true });

  const context = await browser.newContext({
    viewport,
    recordVideo: { dir: videoDir, size: viewport },
  });
  const page = await context.newPage();
  try {
    await run(page);
  } finally {
    // Video is only flushed to disk once the context closes.
    await context.close();
  }

  const files = await fs.readdir(videoDir);
  const webm = files.find((f) => f.endsWith(".webm"));
  if (!webm) throw new Error(`recordFlow(${name}): no .webm produced`);
  return path.join(videoDir, webm);
}

function convertToGif(webmPath, gifPath, { tailSeconds, width = GIF_WIDTH, fps = GIF_FPS } = {}) {
  // Some flows (PlayersPage in particular - see its flow's comment) wait on
  // a backend fetch whose duration varies a lot run to run under concurrent
  // dev-server load. Rather than guess a speedup factor, tailSeconds keeps
  // only the last N seconds of the raw recording - the confirmed-loaded,
  // settled state - so the output is always a short, real demo regardless
  // of how long the wait itself happened to take.
  const seekArgs = tailSeconds ? ["-sseof", `-${tailSeconds}`] : [];
  const filters = `fps=${fps},scale=${width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse`;
  execFileSync("ffmpeg", ["-y", ...seekArgs, "-i", webmPath, "-vf", filters, "-loop", "0", gifPath], {
    stdio: "inherit",
  });
}

async function sizeOf(filePath) {
  const stat = await fs.stat(filePath);
  return stat.size;
}

// --------------------------------------------------------------------------
// Flows
// --------------------------------------------------------------------------

const FLOWS = {
  games: {
    gifName: "frontend-games.gif",
    async run(page) {
      await page.goto(`${BASE}/games`, { waitUntil: "domcontentloaded" });
      // GamesPage has no skeleton of its own (jobs come from App-level
      // state) - a real job card's CardActionArea is the concrete "loaded"
      // signal; GamesPage always renders showAddTile={false}, so every
      // CardActionArea found here is a real job, not the "Add game" tile.
      await page.locator(".MuiCardActionArea-root").first().waitFor({ state: "visible", timeout: 20000 });
      await waitForLoaded(page, { settleMs: 1500 });
    },
  },

  teams: {
    gifName: "frontend-teams.gif",
    async run(page) {
      await page.goto(`${BASE}/teams`, { waitUntil: "domcontentloaded" });
      await page.locator('a[href^="/team?id="]').first().waitFor({ state: "visible", timeout: 20000 });
      await waitForLoaded(page, { settleMs: 1500 });
    },
  },

  players: {
    gifName: "frontend-players.gif",
    tailSeconds: 6,
    async run(page) {
      await page.goto(`${BASE}/players`, { waitUntil: "domcontentloaded" });
      // PlayersPage.tsx's loadIdentifiedPlayers() does Promise.all(getResults
      // + getPlayers) across every completed job before `players` state
      // leaves null (which is what keeps the Skeleton placeholders on
      // screen). With 9 completed jobs against this dev machine's
      // single-worker uvicorn, and other concurrent work on this machine,
      // this has been observed taking anywhere from ~5s to 90s+ run to run
      // - confirmed via a direct DOM probe showing real player cards once
      // `players` resolves, not a rename-induced bug. tailSeconds above
      // keeps only the final settled seconds of whatever this takes.
      await page.locator('a[href^="/player?name="]').first().waitFor({ state: "visible", timeout: 180000 });
      // A big single-shot DOM swap (skeleton -> ~12 real cards) needs more
      // than a brief settle for Chromium's screencast-based video recorder
      // to reliably flush the new frame - too short a wait here previously
      // produced GIFs that captured the skeleton state even though the
      // real DOM (verified separately via page.screenshot) had already
      // loaded correctly.
      await waitForLoaded(page, { settleMs: 4000, timeout: 180000 });
      await page.mouse.wheel(0, 400);
      await page.waitForTimeout(1500);
    },
  },

  "game-results": {
    gifName: "frontend-game-results.gif",
    viewport: RESULTS_VIEWPORT,
    gifWidth: RESULTS_GIF_WIDTH,
    gifFps: RESULTS_GIF_FPS,
    async run(page) {
      await page.goto(`${BASE}/game?job=${RESULTS_JOB_ID}&tab=stats`, { waitUntil: "domcontentloaded" });
      // settleMs is generous (see players flow's comment on why a big
      // single-shot DOM swap needs real time to reach the video recorder).
      await waitForLoaded(page, { settleMs: 2000, timeout: 90000 });
      await tagScrollContainer(page);
      await scrollContentDown(page);

      // Click through the remaining tabs in order, syncing ?tab= via the
      // app's own URL-writing (see ResultsView.writeTabToUrl) - clicking
      // the real Tab buttons is more representative of an actual demo than
      // driving page.goto per tab. Analytics in particular fans out several
      // backend requests (action-quality alone measured ~8s in isolation,
      // more under concurrent load) - give every tab real room to finish
      // rather than capturing a half-loaded skeleton.
      const remainingTabs = [
        { name: "Analytics", exact: true },
        { name: "Rallies", exact: true },
        { name: "Actions", exact: true },
        { name: /Setup/, exact: false },
      ];

      for (const tab of remainingTabs) {
        await page.getByRole("tab", { name: tab.name, exact: tab.exact }).click();
        await waitForLoaded(page, { settleMs: 2000, timeout: 90000 });
        await tagScrollContainer(page);
        await scrollContentDown(page);
      }
    },
  },
};

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

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
      const webmPath = await recordFlow(browser, name, flow.run, { viewport: flow.viewport });
      const gifPath = path.join(OUT_DIR, flow.gifName);
      convertToGif(webmPath, gifPath, { tailSeconds: flow.tailSeconds, width: flow.gifWidth, fps: flow.gifFps });
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
