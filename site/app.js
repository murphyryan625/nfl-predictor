// Shared helpers for both dashboard pages. Plain JavaScript, no build step.

const TZ = "America/Chicago";

async function loadJSON(path) {
  const r = await fetch(path + "?v=" + Date.now());  // bypass stale caches after a new run
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = p => Math.round(p * 100) + "%";
// Fair (no-vig) American moneyline implied by a win probability, e.g. 0.75 -> -300, 0.25 -> +300.
function moneyline(p) {
  p = Math.min(Math.max(p, 0.001), 0.999);
  if (Math.abs(p - 0.5) < 0.0025) return "+100";
  return p > 0.5 ? "−" + Math.round(100 * p / (1 - p)) : "+" + Math.round(100 * (1 - p) / p);
}
const fmtDay = iso => new Date(iso).toLocaleDateString("en-US", { timeZone: TZ, weekday: "long", month: "short", day: "numeric" });
const fmtTime = iso => new Date(iso).toLocaleTimeString("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit" });
const fmtDateTime = iso => new Date(iso).toLocaleString("en-US", { timeZone: TZ, weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

// GitHub Pages URL https://user.github.io/repo/ -> https://github.com/user/repo
function repoUrl() {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  return m && repo ? `https://github.com/${m[1]}/${repo}` : null;
}

// Warn if the scheduled pipeline seems to have stopped (no update in 8+ days).
function showStatus(generatedAt) {
  const url = repoUrl();
  const link = document.getElementById("repo-link");
  if (link && url) link.href = url + "/blob/main/MODEL_REPORT.md";
  const ageDays = (Date.now() - new Date(generatedAt)) / 864e5;
  if (ageDays > 8) {
    const actions = url ? ` <a href="${url}/actions">Check the GitHub Actions runs</a>.` : "";
    document.getElementById("status").innerHTML =
      `<div class="banner">⚠ These predictions haven't been updated in ${Math.floor(ageDays)} days — the scheduled run may be failing.${actions}</div>`;
  }
}

function teamRow(abbr, side, g, teams, prob, isFav, score) {
  const t = teams[abbr] || { name: abbr, color: "#888" };
  const qb = g[side + "_qb"] || "QB unknown";
  const flags = g[side + "_qb_flags"] || [];
  return `<div class="team">
    <span class="chip" style="background:${esc(t.color)}"></span>
    <div><div class="tname">${esc(t.name)}${side === "home" ? '<span class="home">home</span>' : ""}</div>
      <div class="qb ${flags.length ? "flagged" : ""}">QB ${esc(qb)}${flags.length ? " ⚠" : ""}</div></div>
    <div>${score != null ? `<span class="score">${score}</span>` : ""}<span class="prob ${isFav ? "fav" : ""}">${pct(prob)}</span><span class="ml">${moneyline(prob)}</span></div>
  </div>`;
}

function gameCard(g, teams) {
  const pH = g.home_win_prob, pA = 1 - pH;
  const fav = g.pred_margin >= 0 ? g.home_team : g.away_team;
  const favName = (teams[fav] || {}).nick || fav;
  const line = Math.abs(g.pred_margin) < 0.25 ? "Pick'em" : `${esc(favName)} −${Math.abs(g.pred_margin).toFixed(1)}`;
  const done = g.home_score != null;
  let result = "";
  if (done) {
    const margin = g.home_score - g.away_score;
    const right = margin === 0 ? null : (margin > 0) === (pH >= 0.5);
    result = right === null ? `<span>Tie</span>`
      : `<span class="result ${right ? "good" : "bad"}">${right ? "✓ Correct" : "✗ Missed"}</span>`;
  }
  const hc = (teams[g.home_team] || {}).color || "#888", ac = (teams[g.away_team] || {}).color || "#888";
  const notes = (g.notes || []);
  return `<div class="card">
    <div class="card-head"><span>${fmtTime(g.kickoff_ct)} CT${g.game_type !== "REG" ? " · " + esc(g.game_type) : ""}</span>
      <span>${done ? "Final" : ""}</span></div>
    ${teamRow(g.away_team, "away", g, teams, pA, pA > pH, done ? g.away_score : null)}
    ${teamRow(g.home_team, "home", g, teams, pH, pH >= pA, done ? g.home_score : null)}
    <div class="bar"><span style="width:${pA * 100}%;background:${esc(ac)}"></span><span style="width:${pH * 100}%;background:${esc(hc)}"></span></div>
    <div class="spread"><span>Predicted spread: <b>${line}</b> · Fair ML: <b>${esc(favName)} ${moneyline(Math.max(pH, pA))}</b></span>${result}</div>
    ${notes.length ? `<ul class="notes">${notes.map(n => `<li>⚠ ${esc(n)}</li>`).join("")}</ul>` : ""}
  </div>`;
}
