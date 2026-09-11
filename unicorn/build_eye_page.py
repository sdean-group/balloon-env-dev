"""Build the eye-test page: inline the rendered PNGs as data URIs into one HTML file.

    python build_eye_page.py <png dir> <out.html>
"""
import base64, json, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
def uri(p): return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
levels = [("top_53hPa", "53 hPa", "top of the band"), ("bottom_134hPa", "134 hPa", "bottom of the band")]
frames = {k: [uri(src / f"{k}_f{f:02d}.png") for f in range(13)] for k, _, _ in levels}
zoom = {k: uri(src / f"{k}_zoom.png") for k, _, _ in levels}
uwind = {k: uri(src / f"{k}_u.png") for k, _, _ in levels}
total = sum(len(v) for vs in frames.values() for v in vs) + sum(len(v) for v in zoom.values()) + sum(len(v) for v in uwind.values())
print(f"embedded {total / 1e6:.1f} MB of images")

html = """<title>Generated Stratosphere, 10 July 2023</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --ground: #eef1f4; --panel: #ffffff; --ink: #16222c; --muted: #5b6b78; --rule: #cfd7de;
  --accent: #1f5fa8; --accent-ink: #ffffff; --chip: #dfe7ef; --frame: #b9c4cd;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #10171d; --panel: #18222a; --ink: #e6ecf1; --muted: #98a7b3; --rule: #2b3842;
    --accent: #6ea4e6; --accent-ink: #0e1a27; --chip: #24313b; --frame: #3a4956;
  }
}
:root[data-theme="dark"] {
  --ground: #10171d; --panel: #18222a; --ink: #e6ecf1; --muted: #98a7b3; --rule: #2b3842;
  --accent: #6ea4e6; --accent-ink: #0e1a27; --chip: #24313b; --frame: #3a4956;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink); font-family: "Source Serif 4", Georgia, "Times New Roman", serif; font-size: 16.5px; line-height: 1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; display: grid; gap: 28px; }
h1, h2 { font-family: "Archivo Narrow", "Arial Narrow", Arial, sans-serif; font-weight: 600; letter-spacing: 0.005em; text-wrap: balance; margin: 0; }
h1 { font-size: 34px; line-height: 1.1; }
h2 { font-size: 21px; }
.eyebrow { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
header { display: grid; gap: 10px; }
header p { max-width: 68ch; margin: 0; }
.guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px 22px; padding: 16px 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.guide div { font-size: 15px; }
.guide b { font-family: "Archivo Narrow", "Arial Narrow", Arial, sans-serif; font-weight: 600; font-size: 16px; display: block; margin-bottom: 2px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 14px 22px; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.seg button { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; padding: 7px 14px; border: 0; background: var(--panel); color: var(--ink); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--rule); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
.seg button:focus-visible, .play:focus-visible, input[type=range]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.scrub { display: flex; align-items: center; gap: 12px; flex: 1 1 320px; }
.scrub label { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; color: var(--muted); white-space: nowrap; }
.scrub output { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; min-width: 9ch; font-variant-numeric: tabular-nums; }
input[type=range] { flex: 1; accent-color: var(--accent); }
.play { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; padding: 7px 14px; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); color: var(--ink); cursor: pointer; }
figure { margin: 0; display: grid; gap: 8px; }
figure img { width: 100%; height: auto; display: block; background: #fff; border: 1px solid var(--frame); border-radius: 3px; }
figcaption { font-size: 14.5px; color: var(--muted); max-width: 80ch; }
section { display: grid; gap: 12px; }
table { border-collapse: collapse; font-size: 14.5px; width: 100%; max-width: 760px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums; }
th { font-family: "Archivo Narrow", "Arial Narrow", Arial, sans-serif; font-weight: 600; }
td.num, th.num { text-align: right; font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13.5px; }
.tablewrap { overflow-x: auto; }
.note { font-size: 15px; color: var(--muted); max-width: 72ch; margin: 0; }
@media (prefers-reduced-motion: reduce) { .play { display: none; } }
</style>
<main>
  <header>
    <div class="eyebrow">held-out day &middot; ERA5 model levels 49&ndash;66 (about 53 to 134 hPa) &middot; HEALPix nside 256, 0.23&deg;</div>
    <h1>Generated Stratosphere, 10 July 2023</h1>
    <p>Wind speed over the globe for thirteen hours, three ways: what ERA5 recorded, what the fine model produces when it is handed ERA5's own coarse field, and what the full generator produces on its own for that date. The held-out week was never used in training.</p>
  </header>

  <div class="guide">
    <div><b>ERA5 fine</b> The reanalysis at 0.23&deg;. The reference.</div>
    <div><b>Stage 2 given ERA5's coarse field</b> Same weather as ERA5: the fine model rebuilds the detail from ERA5's 1.8&deg; block means. Compare these two panels pixel for pixel.</div>
    <div><b>Stage 1 &rarr; Stage 2</b> A free sample for this date and QBO phase. Different weather by design, so judge it on character: where the jets sit, how strong, how much texture.</div>
    <div><b>Stage 1 coarse field</b> The 1.8&deg; field the free sample was built on, interpolated between its 6-hourly frames.</div>
  </div>

  <section>
    <div class="controls">
      <div class="seg" role="group" aria-label="pressure level">
        __LEVEL_BUTTONS__
      </div>
      <div class="scrub">
        <label for="hour">hour (UTC)</label>
        <input id="hour" type="range" min="0" max="12" value="6" step="1" aria-label="hour of the block">
        <output id="hourout" for="hour">06:00</output>
      </div>
      <button class="play" id="play" type="button">Play 13 hours</button>
    </div>
    <figure>
      <img id="frame" alt="Four wind-speed maps of the globe for the selected level and hour: ERA5, Stage 2 given ERA5's coarse field, the full generator, and its coarse field.">
      <figcaption>Colour is wind speed in m/s on one scale per level, clipped at the 99.5th percentile of ERA5 at hour 6. Coarse frames exist at hours 0, 6 and 12; the fine model invents the hours between.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Native resolution, hour 6</h2>
    <figure>
      <img id="zoom" alt="Zoomed wind-speed maps over the southern ocean at native resolution: ERA5, Stage 2 given ERA5's coarse field, and the coarse field alone.">
      <figcaption>A 40&deg; by 50&deg; window over the southern ocean (70&deg;S to 30&deg;S, 0&deg; to 50&deg;E), the edge of the winter polar vortex. Left is ERA5, middle is the fine model given ERA5's coarse field, right is that coarse field by itself: the difference between the right and middle panels is what Stage 2 adds.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Zonal wind, hour 6</h2>
    <figure>
      <img id="uwind" alt="Three maps of the eastward wind component: ERA5, Stage 2 given ERA5's coarse field, and the full generator.">
      <figcaption>Eastward wind u, red eastward and blue westward. Sign is what a balloon feels; in July the southern winter vortex is a ring of strong westerlies while the summer hemisphere is easterly.</figcaption>
    </figure>
  </section>

  <section>
    <h2>What the gate measured for this model</h2>
    <p class="note">The smooth-baseline Stage 2 at 200k steps (the model of record), scored on four held-out 13-hour blocks (one per season) with Stage 1 supplying the coarse field, which is how the simulator will use it.</p>
    <div class="tablewrap">
      <table>
        <thead><tr><th>quantity</th><th class="num">ERA5 / floor</th><th class="num">generated</th></tr></thead>
        <tbody>
          <tr><td>spectrum, coarse band (l 10&ndash;96), log ratio</td><td class="num">&plusmn;0.12</td><td class="num">+0.01</td></tr>
          <tr><td>spectrum, 160&ndash;400 km band, log ratio</td><td class="num">&plusmn;0.08</td><td class="num">&minus;0.13</td></tr>
          <tr><td>fine-scale residual, RMS (m/s)</td><td class="num">1.11</td><td class="num">1.07</td></tr>
          <tr><td>residual persistence, 1 / 3 / 6 h correlation</td><td class="num">0.91 / 0.64 / 0.45</td><td class="num">0.91 / 0.63 / 0.43</td></tr>
          <tr><td>W1 distance of u / v (m/s), floor 2.78</td><td class="num">&mdash;</td><td class="num">0.56 / 0.12</td></tr>
          <tr><td>99.9th percentile wind speed (m/s)</td><td class="num">62.6</td><td class="num">60.1</td></tr>
          <tr><td>columns with opposing winds between levels</td><td class="num">30.0%</td><td class="num">31.3%</td></tr>
          <tr><td>jump across face edges, relative to ERA5</td><td class="num">1.00</td><td class="num">1.02</td></tr>
          <tr><td>block means equal the coarse field</td><td class="num">exact</td><td class="num">exact</td></tr>
        </tbody>
      </table>
    </div>
    <p class="note">The poster-style benchmark (summer suite on its NE Pacific window) is in the meeting notes.</p>
  </section>
</main>
<script>
const FRAMES = __FRAMES__;
const ZOOM = __ZOOM__;
const UWIND = __UWIND__;
const LEVELS = __LEVELS__;
let level = LEVELS[0][0], hour = 6, timer = null;
const img = document.getElementById("frame"), zoom = document.getElementById("zoom"), uw = document.getElementById("uwind");
const range = document.getElementById("hour"), out = document.getElementById("hourout"), play = document.getElementById("play");
function show() {
  img.src = FRAMES[level][hour]; zoom.src = ZOOM[level]; uw.src = UWIND[level];
  out.value = String(hour).padStart(2, "0") + ":00"; range.value = hour;
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.level === level ? "true" : "false"));
}
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { level = b.dataset.level; show(); }));
range.addEventListener("input", () => { hour = Number(range.value); show(); });
function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = "Play 13 hours"; } }
play.addEventListener("click", () => {
  if (timer) { stop(); return; }
  play.textContent = "Pause";
  timer = setInterval(() => { hour = (hour + 1) % 13; show(); }, 550);
});
document.addEventListener("keydown", e => {
  if (e.target === range) return;
  if (e.key === "ArrowRight") { hour = Math.min(12, hour + 1); show(); }
  if (e.key === "ArrowLeft") { hour = Math.max(0, hour - 1); show(); }
});
try { const saved = localStorage.getItem("eye-level"); if (saved && FRAMES[saved]) level = saved; } catch (e) {}
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { try { localStorage.setItem("eye-level", level); } catch (e) {} }));
show();
</script>
"""
buttons = "\n        ".join(f'<button type="button" data-level="{k}" aria-pressed="{"true" if i == 0 else "false"}">{lab} &middot; {desc}</button>' for i, (k, lab, desc) in enumerate(levels))
html = (html.replace("__LEVEL_BUTTONS__", buttons).replace("__FRAMES__", json.dumps(frames)).replace("__ZOOM__", json.dumps(zoom))
        .replace("__UWIND__", json.dumps(uwind)).replace("__LEVELS__", json.dumps([[k, lab] for k, lab, _ in levels])))
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
