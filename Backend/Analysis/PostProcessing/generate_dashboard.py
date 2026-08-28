import json
from pathlib import Path

STATS_LOG_NAME = "player_stats.json"
GAME_STATUS_LOG_NAME = "game_status.json"
DASHBOARD_NAME = "dashboard.html"

ACTION_TYPES = ["serve", "spike", "set", "dig", "block", "hit"]

TEMPLATE = r"""<title>Touch Count</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

  :root {
    --ground:        #f7f3ec;
    --surface:        #fffdf9;
    --surface-raised: #ffffff;
    --ink-primary:    #211d17;
    --ink-secondary:  #635a4c;
    --ink-muted:      #9a8f7c;
    --border:         rgba(33,29,23,0.10);
    --accent:         #b5772a;
    --accent-ink:     #ffffff;
    --status-live:    #0ca30c;
    --status-dead:    #9a8f7c;

    --cat-serve: #2a78d6;
    --cat-spike: #eb6834;
    --cat-set:   #1baf7a;
    --cat-dig:   #eda100;
    --cat-block: #e87ba4;
    --cat-hit:   #008300;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground:        #14120f;
      --surface:        #1c1914;
      --surface-raised: #211d17;
      --ink-primary:    #f5f1ea;
      --ink-secondary:  #c3b9a8;
      --ink-muted:      #8a8071;
      --border:         rgba(245,241,234,0.12);
      --accent:         #d99a4e;
      --accent-ink:     #1c1914;
      --status-live:    #4fd44f;
      --status-dead:    #8a8071;

      --cat-serve: #3987e5;
      --cat-spike: #d95926;
      --cat-set:   #199e70;
      --cat-dig:   #c98500;
      --cat-block: #d55181;
      --cat-hit:   #2fae2f;
    }
  }
  :root[data-theme="dark"] {
    --ground:        #14120f;
    --surface:        #1c1914;
    --surface-raised: #211d17;
    --ink-primary:    #f5f1ea;
    --ink-secondary:  #c3b9a8;
    --ink-muted:      #8a8071;
    --border:         rgba(245,241,234,0.12);
    --accent:         #d99a4e;
    --accent-ink:     #1c1914;
    --status-live:    #4fd44f;
    --status-dead:    #8a8071;

    --cat-serve: #3987e5;
    --cat-spike: #d95926;
    --cat-set:   #199e70;
    --cat-dig:   #c98500;
    --cat-block: #d55181;
    --cat-hit:   #2fae2f;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink-primary);
    font-family: "Source Sans 3", system-ui, -apple-system, sans-serif;
    line-height: 1.5;
  }
  .wrap { max-width: 920px; margin: 0 auto; padding: 40px 20px 80px; }

  .eyebrow {
    font-family: "Source Sans 3", sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 6px;
  }
  h1 {
    font-family: "Oswald", system-ui, sans-serif;
    font-weight: 700;
    font-size: clamp(32px, 5vw, 44px);
    letter-spacing: 0.01em;
    margin: 0 0 28px;
    text-wrap: balance;
  }

  .summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 36px;
  }
  .stat-tile {
    background: var(--surface);
    padding: 18px 20px;
  }
  .stat-tile .value {
    font-family: "Oswald", sans-serif;
    font-size: 30px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
  }
  .stat-tile .label {
    font-size: 12.5px;
    color: var(--ink-secondary);
    margin-top: 4px;
  }

  .section-title {
    font-family: "Oswald", sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--ink-secondary);
    margin: 0 0 12px;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 18px;
    font-size: 13px;
    color: var(--ink-secondary);
  }
  .legend .swatch {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 2px;
    margin-right: 6px;
    vertical-align: -1px;
  }

  .roster { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface); }

  .roster-header, .player-row {
    display: grid;
    grid-template-columns: 56px 1fr 190px 80px 120px 20px;
    align-items: center;
    gap: 14px;
    padding: 12px 18px;
  }
  .roster-header {
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-muted);
    border-bottom: 1px solid var(--border);
    user-select: none;
  }
  .roster-header button {
    all: unset;
    cursor: pointer;
    color: inherit;
    font: inherit;
    letter-spacing: inherit;
    text-transform: inherit;
  }
  .roster-header button:hover, .roster-header button:focus-visible { color: var(--accent); }
  .roster-header button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }
  .roster-header .active { color: var(--accent); }

  .player-row {
    border-bottom: 1px solid var(--border);
    cursor: pointer;
  }
  .player-row:last-child { border-bottom: none; }
  .player-row:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
  .player-row:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }

  .jersey {
    width: 38px; height: 38px;
    border-radius: 8px;
    background: var(--surface-raised);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-family: "Oswald", sans-serif;
    font-weight: 600;
    font-size: 15px;
    font-variant-numeric: tabular-nums;
  }

  .hits-total {
    font-family: "Oswald", sans-serif;
    font-size: 20px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .breakdown-bar {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--border);
  }
  .breakdown-bar span { display: block; height: 100%; }
  .breakdown-bar span + span { border-left: 2px solid var(--surface); }

  .rallies-cell { font-variant-numeric: tabular-nums; color: var(--ink-secondary); font-size: 14px; }

  .chevron {
    transition: transform 0.15s ease;
    color: var(--ink-muted);
    font-size: 12px;
  }
  .player-row[aria-expanded="true"] .chevron { transform: rotate(90deg); }

  .detail {
    display: none;
    padding: 4px 18px 18px 90px;
    border-bottom: 1px solid var(--border);
    background: color-mix(in srgb, var(--accent) 3%, var(--surface));
  }
  .detail.open { display: block; }
  .detail table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  .detail th {
    text-align: left;
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-muted);
    font-weight: 600;
    padding: 6px 10px 6px 0;
  }
  .detail td { padding: 5px 10px 5px 0; }
  .detail td.ts { font-variant-numeric: tabular-nums; color: var(--ink-secondary); }
  .rate-note { font-size: 12.5px; color: var(--ink-muted); margin-top: 10px; }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--ink-primary);
    background: var(--surface-raised);
    border: 1px solid var(--border);
  }
  .chip .dot { width: 7px; height: 7px; border-radius: 50%; }

  details.caveats {
    margin-top: 32px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 4px 18px;
  }
  details.caveats summary {
    cursor: pointer;
    padding: 12px 0;
    font-weight: 600;
    font-size: 14px;
    color: var(--ink-secondary);
  }
  details.caveats summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  details.caveats ul { margin: 0 0 16px; padding-left: 20px; color: var(--ink-secondary); font-size: 13.5px; }
  details.caveats li { margin-bottom: 10px; }

  @media (max-width: 620px) {
    .roster-header, .player-row { grid-template-columns: 44px 1fr 60px 20px; }
    .roster-header span:nth-child(3), .roster-header span:nth-child(5),
    .player-row .breakdown-cell, .player-row .rallies-cell { display: none; }
    .detail { padding-left: 18px; }
  }
</style>

<div class="wrap">
  <p class="eyebrow">Post-Match Analysis</p>
  <h1>Touch Count</h1>

  <div class="summary" id="summary"></div>

  <p class="section-title">Action mix</p>
  <div class="legend" id="legend"></div>

  <p class="section-title">Players</p>
  <div class="roster">
    <div class="roster-header">
      <span></span>
      <button data-sort="id">Player</button>
      <button data-sort="hits" class="active">Touches ▾</button>
      <span>Breakdown</span>
      <button data-sort="rallies">Rallies</button>
      <span></span>
    </div>
    <div id="roster-body"></div>
  </div>

  <details class="caveats">
    <summary>How to read these numbers</summary>
    <ul id="caveats-list"></ul>
  </details>
</div>

<script id="dashboard-data" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('dashboard-data').textContent);
  const ACTION_TYPES = ["serve", "spike", "set", "dig", "block", "hit"];
  const ACTION_COLOR = {
    serve: 'var(--cat-serve)', spike: 'var(--cat-spike)', set: 'var(--cat-set)',
    dig: 'var(--cat-dig)', block: 'var(--cat-block)', hit: 'var(--cat-hit)'
  };
  const ACTION_LABEL = {
    serve: 'Serve', spike: 'Spike', set: 'Set', dig: 'Dig', block: 'Block', hit: 'Hit (unclassified)'
  };

  const players = Object.entries(data.players).map(([id, p]) => ({ id: Number(id), ...p }));

  // --- summary strip ---
  const totalHits = players.reduce((s, p) => s + p.total_hits, 0);
  const rallyCount = data.rally_count;
  const avgDuration = data.avg_rally_duration_s;

  const summaryEl = document.getElementById('summary');
  const tiles = [
    [String(rallyCount), 'Rallies detected'],
    [String(players.length), 'Players credited'],
    [String(totalHits), 'Total touches'],
    [avgDuration != null ? avgDuration.toFixed(1) + 's' : '—', 'Avg rally length']
  ];
  summaryEl.innerHTML = tiles.map(([v, l]) =>
    `<div class="stat-tile"><div class="value">${v}</div><div class="label">${l}</div></div>`
  ).join('');

  // --- legend ---
  document.getElementById('legend').innerHTML = ACTION_TYPES.map(t =>
    `<span><span class="swatch" style="background:${ACTION_COLOR[t]}"></span>${ACTION_LABEL[t]}</span>`
  ).join('');

  // --- roster ---
  const bodyEl = document.getElementById('roster-body');
  let sortKey = 'hits';
  let sortDir = -1;

  function sortValue(p, key) {
    if (key === 'id') return p.id;
    if (key === 'rallies') return p.rallies_participated;
    return p.total_hits;
  }

  function render() {
    const sorted = [...players].sort((a, b) => sortDir * (sortValue(a, sortKey) - sortValue(b, sortKey)));

    bodyEl.innerHTML = sorted.map(p => {
      const bar = ACTION_TYPES
        .filter(t => p.hits_by_type[t] > 0)
        .map(t => `<span style="width:${(p.hits_by_type[t] / p.total_hits * 100).toFixed(1)}%;background:${ACTION_COLOR[t]}" title="${ACTION_LABEL[t]}: ${p.hits_by_type[t]}"></span>`)
        .join('');

      const rows = p.events.map(e => `
        <tr>
          <td class="ts">${e.timestamp_s.toFixed(1)}s</td>
          <td><span class="chip"><span class="dot" style="background:${ACTION_COLOR[e.action_type]}"></span>${ACTION_LABEL[e.action_type]}</span></td>
          <td>${e.rally_index != null ? 'Rally ' + e.rally_index : '—'}</td>
        </tr>`).join('');

      const rateNote = p.rally_ending_touch_rate != null
        ? `Last touch in ${(p.rally_ending_touch_rate * 100).toFixed(0)}% of rallies played (${p.rally_ending_touches}/${p.rallies_participated}) — see caveats below before reading this as an error rate.`
        : '';

      return `
        <div class="player-row" role="button" tabindex="0" aria-expanded="false" data-id="${p.id}">
          <div class="jersey">#${p.id}</div>
          <div>Player ${p.id}</div>
          <div class="breakdown-cell"><div class="breakdown-bar">${bar}</div></div>
          <div class="hits-total">${p.total_hits}</div>
          <div class="rallies-cell">${p.rallies_participated} rallies</div>
          <div class="chevron">▶</div>
        </div>
        <div class="detail" data-detail-for="${p.id}">
          <table>
            <thead><tr><th>Time</th><th>Action</th><th>Rally</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
          ${rateNote ? `<p class="rate-note">${rateNote}</p>` : ''}
        </div>`;
    }).join('');

    bodyEl.querySelectorAll('.player-row').forEach(row => {
      const toggle = () => {
        const detail = bodyEl.querySelector(`.detail[data-detail-for="${row.dataset.id}"]`);
        const isOpen = detail.classList.toggle('open');
        row.setAttribute('aria-expanded', String(isOpen));
      };
      row.addEventListener('click', toggle);
      row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    });
  }

  document.querySelectorAll('.roster-header button').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.sort;
      if (sortKey === key) { sortDir *= -1; } else { sortKey = key; sortDir = -1; }
      document.querySelectorAll('.roster-header button').forEach(b => {
        b.classList.remove('active');
        b.textContent = b.textContent.replace(' ▾', '').replace(' ▴', '');
      });
      btn.classList.add('active');
      btn.textContent += sortDir === -1 ? ' ▾' : ' ▴';
      render();
    });
  });

  document.getElementById('caveats-list').innerHTML =
    data.caveats.map(c => `<li>${c}</li>`).join('');

  render();
})();
</script>
"""


def generateDashboard(output_path):
    """
    Builds a self-contained, interactive HTML dashboard from player_stats.json
    (sortable roster, per-player action-type breakdown, expandable event
    timelines) - open dashboard.html in any browser, no server needed.
    """
    output_path = Path(output_path)
    stats_file = output_path / STATS_LOG_NAME

    if not stats_file.exists():
        print(f"No player stats found at {stats_file}; run consolidation first.")
        return

    with open(stats_file) as f:
        stats = json.load(f)

    rally_count = None
    avg_rally_duration_s = None
    status_file = output_path / GAME_STATUS_LOG_NAME
    if status_file.exists():
        with open(status_file) as f:
            rallies = json.load(f)["rallies"]
        rally_count = len(rallies)
        if rallies:
            avg_rally_duration_s = sum(r["duration_s"] for r in rallies) / len(rallies)

    payload = {
        "players": stats["players"],
        "caveats": stats["caveats"],
        "rally_count": rally_count if rally_count is not None else len(
            {e["rally_index"] for p in stats["players"].values() for e in p["events"] if e["rally_index"] is not None}
        ),
        "avg_rally_duration_s": avg_rally_duration_s
    }

    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))

    dashboard_file = output_path / DASHBOARD_NAME
    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved: {dashboard_file}")
