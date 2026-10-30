"""
viz.py - animated "living map" of the simulator, drawn as SVG with SMIL animations
(no JavaScript needed). One scene is rendered per round and streamed to the page.

    scene_html(mode, ...) -> HTML string

Visual language:
  blue packet  = global model sent DOWN to a junction
  amber packet = update sent UP to the server (smaller when INT8-compressed)
  red dots     = raw records uploaded to the cloud (centralised mode)
  grey + zzz   = junction asleep this round (PulseFed); amber ring = forced back in
"""
import random

W, H = 960, 540
JX = [150, 370, 590, 810]          # junction x positions
JY = 405                           # junction y position (main road)
SX, SY = 480, 78                   # server centre
LINK_TOP = SY + 34
CAR_COLORS = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6"]
JSUB_NONIID = ["quietest roads", "quiet–medium roads", "medium–busy roads", "busiest roads"]


def _link(j, reverse=False):
    x = JX[j]
    if reverse:
        return f"M {SX} {LINK_TOP} Q {x} {LINK_TOP + 70} {x} {JY - 72}"
    return f"M {x} {JY - 72} Q {x} {LINK_TOP + 70} {SX} {LINK_TOP}"


def _cars(rng, path, n, dur, flip=False):
    out = []
    for k in range(n):
        c = rng.choice(CAR_COLORS)
        begin = -dur * k / max(n, 1) - rng.random() * 0.3
        out.append(
            f'<rect x="-7" y="-4" width="14" height="8" rx="2.5" fill="{c}" stroke="#111827" stroke-width="0.6">'
            f'<animateMotion path="{path}" dur="{dur:.2f}s" begin="{begin:.2f}s" repeatCount="indefinite" rotate="auto"/>'
            f"</rect>")
    return "".join(out)


def _roads(rng, density):
    s = [f'<rect x="0" y="{JY - 26}" width="{W}" height="52" fill="#4b5563"/>',
         f'<line x1="0" y1="{JY}" x2="{W}" y2="{JY}" stroke="#f9fafb" stroke-width="2" stroke-dasharray="14 12"/>']
    for x in JX:
        s.append(f'<rect x="{x - 22}" y="{JY - 75}" width="44" height="{H - JY + 75}" fill="#4b5563"/>')
        s.append(f'<line x1="{x}" y1="{JY - 75}" x2="{x}" y2="{H}" stroke="#f9fafb" stroke-width="2" stroke-dasharray="10 10"/>')
    # cars on the main road (both directions)
    s.append(_cars(rng, f"M -20 {JY + 13} L {W + 20} {JY + 13}", 5, 9))
    s.append(_cars(rng, f"M {W + 20} {JY - 13} L -20 {JY - 13}", 5, 10))
    # cars on each junction's side road: busier junction -> more cars
    for j, x in enumerate(JX):
        n = density[j]
        s.append(_cars(rng, f"M {x + 11} {JY - 75} L {x + 11} {H + 10}", n, 4.2))
        s.append(_cars(rng, f"M {x - 11} {H + 10} L {x - 11} {JY - 75}", n, 4.6))
    return "".join(s)


def _traffic_light(x, y, state):
    lit = {"talk": (2, "#22c55e"), "sleep": (0, "#ef4444"), "forced": (1, "#f59e0b"), "idle": (2, "#22c55e")}[state]
    s = [f'<rect x="{x - 9}" y="{y - 27}" width="18" height="46" rx="5" fill="#111827"/>']
    for i, cy in enumerate([y - 17, y - 3, y + 11]):
        col = lit[1] if i == lit[0] else "#374151"
        s.append(f'<circle cx="{x}" cy="{cy}" r="5" fill="{col}"/>')
    return "".join(s)


def _packet(path_d, color, size, phase, label=None):
    """phase 0 = first half of a 2 s cycle (downlink), 1 = second half (uplink)."""
    kp, kt = ("0;1;1", "0;0.5;1") if phase == 0 else ("0;0;1", "0;0.5;1")
    vis = "1;0" if phase == 0 else "0;1"
    h = size / 2
    txt = (f'<text x="0" y="{-h - 4}" font-size="10" text-anchor="middle" fill="#111827" '
           f'font-family="Calibri,Arial">{label}</text>') if label else ""
    return (f'<g opacity="{1 if phase == 0 else 0}">'
            f'<rect x="{-h}" y="{-h}" width="{size}" height="{size}" rx="3" fill="{color}" stroke="#fff" stroke-width="1.5"/>{txt}'
            f'<animateMotion path="{path_d}" dur="2s" repeatCount="indefinite" keyPoints="{kp}" keyTimes="{kt}" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="{vis}" keyTimes="0;0.5" dur="2s" repeatCount="indefinite" calcMode="discrete"/>'
            f"</g>")


def _raw_stream(j):
    out = []
    for k in range(4):
        out.append(f'<circle r="4.5" fill="#dc2626" stroke="#fff" stroke-width="1">'
                   f'<animateMotion path="{_link(j)}" dur="1.6s" begin="{-0.4 * k:.1f}s" repeatCount="indefinite"/></circle>')
    return "".join(out)


def _badge_color(v, lo, hi):
    t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
    r = int(34 + t * (220 - 34)); g = int(197 - t * (197 - 38)); b = int(94 - t * (94 - 38))
    return f"#{r:02x}{g:02x}{b:02x}"


def scene_html(mode, title, round_no=0, total_rounds=1, active=None, forced=None, mae=None,
               mb=None, down_kb=None, up_kb=None, density=(1, 2, 3, 4), per_client=None,
               raw_mb=None, finished=False, seed=0, iid=False):
    """
    mode: "centralized" | "federated"
    active/forced: junction ids talking this round / forced back in (federated)
    per_client: final MAE per junction (shown as badges at the end)
    """
    rng = random.Random(1000 * seed + round_no)
    active = list(range(4)) if active is None else list(active)
    forced = forced or []
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'style="background:#eef6ee;border-radius:12px;font-family:Calibri,Arial,sans-serif;max-width:1000px;display:block;margin:0 auto">']
    # sky band + server
    p.append(f'<rect x="0" y="0" width="{W}" height="200" fill="#e8f1fb"/>')
    p.append(_roads(rng, density))
    p.append(f'<rect x="{SX - 105}" y="{SY - 34}" width="210" height="68" rx="16" fill="#14213d"/>')
    p.append(f'<text x="{SX}" y="{SY - 6}" font-size="17" font-weight="700" fill="#fff" text-anchor="middle">'
             f'{"☁ Cloud" if mode == "centralized" else "☁ Aggregation server"}</text>')
    sub = ("trains on pooled raw data" if mode == "centralized"
           else ("averaging updates (FedAvg)" if not finished else "global model ready"))
    p.append(f'<text x="{SX}" y="{SY + 16}" font-size="12" fill="#c7d2fe" text-anchor="middle">{sub}</text>')

    # links + packets
    for j in range(4):
        talking = mode == "centralized" or j in active
        dash = "" if talking else ' stroke-dasharray="6 7"'
        col = "#94a3b8" if talking else "#cbd5e1"
        p.append(f'<path d="{_link(j)}" fill="none" stroke="{col}" stroke-width="2.5"{dash}/>')
        if finished:
            continue
        if mode == "centralized":
            p.append(_raw_stream(j))
        elif talking:
            p.append(_packet(_link(j, reverse=True), "#2563eb", 15, 0, f"{down_kb:.1f} KB" if j == 0 and down_kb else None))
            usize = 15 if (up_kb or 18) > 10 else 9
            p.append(_packet(_link(j), "#f59e0b", usize, 1, f"{up_kb:.1f} KB" if j == 0 and up_kb else None))

    # junctions
    lo = min(per_client) if per_client else 0
    hi = max(per_client) if per_client else 1
    for j, x in enumerate(JX):
        talking = mode == "centralized" or j in active
        state = "idle" if finished else ("forced" if j in forced else ("talk" if talking else "sleep"))
        box = "#ffffff" if talking or finished else "#e5e7eb"
        ink = "#14213d" if talking or finished else "#9ca3af"
        p.append(f'<rect x="{x - 78}" y="{JY - 112}" width="156" height="42" rx="9" fill="{box}" '
                 f'stroke="#14213d" stroke-width="1.5"/>')
        p.append(f'<text x="{x}" y="{JY - 94}" font-size="13" font-weight="700" text-anchor="middle" fill="{ink}">Junction {j}</text>')
        sub = "mixed traffic (IID)" if iid else JSUB_NONIID[j]
        p.append(f'<text x="{x}" y="{JY - 78}" font-size="11" text-anchor="middle" fill="{ink}">{sub}</text>')
        p.append(_traffic_light(x + 94, JY - 92, state))
        if j in forced and not finished:
            p.append(f'<rect x="{x - 84}" y="{JY - 118}" width="168" height="54" rx="12" fill="none" '
                     f'stroke="#f59e0b" stroke-width="3">'
                     f'<animate attributeName="opacity" values="1;0.15;1" dur="1s" repeatCount="indefinite"/></rect>')
        if state == "sleep":
            p.append(f'<text x="{x - 30}" y="{JY - 120}" font-size="16" font-weight="700" fill="#64748b">z z z'
                     f'<animate attributeName="opacity" values="0.2;1;0.2" dur="1.6s" repeatCount="indefinite"/></text>')
        if per_client:
            v = per_client[j]
            p.append(f'<rect x="{x - 55}" y="{JY + 60}" width="110" height="26" rx="13" fill="{_badge_color(v, lo, hi)}"/>')
            p.append(f'<text x="{x}" y="{JY + 78}" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">MAE {v:.4f}</text>')

    # status panel
    lines = [f'<tspan x="22" dy="0" font-size="16" font-weight="700" fill="#14213d">{title}</tspan>']
    lines.append(f'<tspan x="22" dy="22" font-size="13" fill="#1f2937">'
                 f'{"Epoch" if mode == "centralized" else "Round"} {round_no} / {total_rounds}</tspan>')
    if mae is not None:
        lines.append(f'<tspan x="22" dy="19" font-size="13" fill="#1f2937">Test MAE <tspan font-weight="700">{mae:.4f}</tspan></tspan>')
    if mode == "centralized" and raw_mb is not None:
        lines.append(f'<tspan x="22" dy="19" font-size="13" fill="#1f2937">Raw data uploaded once: <tspan font-weight="700">{raw_mb:.2f} MB</tspan></tspan>')
    elif mb is not None:
        lines.append(f'<tspan x="22" dy="19" font-size="13" fill="#1f2937">Data sent so far: <tspan font-weight="700">{mb:.2f} MB</tspan></tspan>')
        if not finished:
            lines.append(f'<tspan x="22" dy="19" font-size="12" fill="#475569">{len(active)} of 4 junctions talking this round</tspan>')
    p.append(f'<text x="22" y="30">{"".join(lines)}</text>')

    # legend
    lg = [("#2563eb", "model down"), ("#f59e0b", "update up"), ("#dc2626", "raw data (centralised)")]
    for i, (c, t) in enumerate(lg):
        y = 26 + i * 20
        p.append(f'<rect x="{W - 190}" y="{y - 10}" width="12" height="12" rx="2" fill="{c}"/>'
                 f'<text x="{W - 172}" y="{y}" font-size="12" fill="#1f2937">{t}</text>')
    p.append(f'<text x="{W - 190}" y="{26 + 3 * 20}" font-size="12" fill="#64748b">grey + zzz = asleep</text>')

    # progress bar
    frac = max(0.0, min(1.0, round_no / max(total_rounds, 1)))
    p.append(f'<rect x="20" y="{H - 14}" width="{W - 40}" height="6" rx="3" fill="#cbd5e1"/>')
    p.append(f'<rect x="20" y="{H - 14}" width="{(W - 40) * frac:.0f}" height="6" rx="3" fill="#2a9d8f"/>')
    p.append("</svg>")
    return "".join(p)
