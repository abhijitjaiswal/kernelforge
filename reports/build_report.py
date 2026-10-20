"""Build reports/klein-m5-report.pdf from recorded runs only (reports/report_data.py).

    python reports/build_report.py
Requires matplotlib + reportlab + pillow. Charts are written to reports/fig/ at 220 dpi.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from report_data import collect, experiment_rows  # noqa: E402

FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
D = collect()

# ------------------------------------------------------------------ fonts / styles
SUP = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Serif", SUP + "Georgia.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Bold", SUP + "Georgia Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Italic", SUP + "Georgia Italic.ttf"))
pdfmetrics.registerFont(TTFont("Sans", SUP + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", SUP + "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Uni", SUP + "Arial Unicode.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-Bold")
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
ARW = '<font name="Uni">→</font>'
INK = colors.HexColor("#1f1f1f")
MUTE = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#bdbdbd")

S = {
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Serif", fontSize=8.2, leading=11, textColor=MUTE, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="Serif-Italic", fontSize=8.2, leading=11, textColor=MUTE, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.2, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=20, leading=25, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=9.5, leading=13, textColor=MUTE, spaceAfter=10),
    "quote": ParagraphStyle("quote", fontName="Serif-Bold", fontSize=10.6, leading=15, textColor=INK, leftIndent=10,
                            borderPadding=(6, 6, 6, 8), borderColor=RULE, borderWidth=0.6, spaceBefore=4, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=8.4, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.2, leading=10.6, textColor=INK),
    "tiny": ParagraphStyle("tiny", fontName="Sans", fontSize=6.3, leading=7.6, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def plain(t):
    return P(f'<font name="Sans-Bold">In plain terms.</font> {t}')


def precise(t):
    return P(f'<font name="Sans-Bold">Precisely.</font> {t}')


def table(rows, widths, header=True, size="cell", zebra=False):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if (header and i == 0) else size]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
          ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, INK))
    for i in range(1, len(rows)):
        st.append(("LINEBELOW", (0, i), (-1, i), 0.25, RULE))
        if zebra and i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f4f2")))
    t.setStyle(TableStyle(st))
    return t


def fig(path, width_cm, caption):
    im = Image.open(path)
    w = width_cm * cm
    h = w * im.size[1] / im.size[0]
    return KeepTogether([RLImage(str(path), width=w, height=h), Spacer(1, 2), P(caption, "cap")])


# ------------------------------------------------------------------ charts
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "gray": "#a3a19a", "ink": "#2b2b2b", "mute": "#6b6a66", "grid": "#e4e3de"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5,
                     "axes.edgecolor": C["mute"], "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})


def chart_waterfall():
    ref_label, ref_val, ref_exp = D["waterfall_ref"]
    steps = D["waterfall"]
    labels = [f"{ref_label}\n{ref_exp} · reference"] + [f"{s[0]}\n{s[2]} · Tier {s[3]}" for s in steps]
    vals = [ref_val] + [s[1] for s in steps]
    fig_, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(vals))
    cols = ["white"] + [C["blue"]] * (len(steps) - 1) + [C["aqua"]]
    bars = ax.bar(x, vals, color=cols, edgecolor=[C["mute"]] + cols[1:], width=0.62, linewidth=0.9)
    bars[0].set_hatch("////")
    for i in range(1, len(vals) - 1):
        ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [vals[i + 1], vals[i + 1]], color=C["mute"], lw=0.6, ls=":")
    for i, (v, b) in enumerate(zip(vals, bars)):
        swap = "\n(swap grew)" if (i and steps[i - 1][4]) or (i == 0) else ""
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f} GB{swap}", ha="center", va="bottom", fontsize=8, color=C["ink"],
                fontweight="bold", linespacing=1.1)
    ax.axhline(8, color=C["orange"], lw=1.1, ls="--")
    ax.text(len(vals) - 0.55, 8.15, "8 GB", color=C["orange"], fontsize=7.5, ha="right", va="bottom")
    ax.axhline(16, color=C["mute"], lw=0.8, ls="-.")
    ax.text(len(vals) - 0.55, 15.65, "16 GB (this machine's total RAM)", color=C["mute"], fontsize=7, ha="right", va="top")
    ax.set_xticks(x, labels, fontsize=6.6)
    ax.set_ylabel("Peak memory of the process (GB, macOS)")
    ax.set_ylim(0, 17)
    ax.axvline(0.5, color=C["grid"], lw=1.2)
    p = FIG / "waterfall.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_stages():
    names = list(D["stage_pairs"])
    before = [D["stage_pairs"][n][0] for n in names]
    after = [D["stage_pairs"][n][1] for n in names]
    fig_, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(names))
    ax.barh(y + 0.19, before, height=0.36, color=C["gray"], label="before")
    ax.barh(y - 0.19, after, height=0.36, color=C["blue"], label="after (exact)")
    for i in range(len(names)):
        ax.text(before[i] + 0.1, y[i] + 0.19, f"{before[i]:.2f} GB", va="center", fontsize=7.6, color=C["ink"])
        ax.text(after[i] + 0.1, y[i] - 0.19, f"{after[i]:.2f} GB  ({100 * (1 - after[i] / before[i]):.0f}% less)", va="center", fontsize=7.6,
                color=C["ink"], fontweight="bold")
    ax.set_yticks(y, [f"{n}\n({D['stage_pairs'][n][2]})" for n in names], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("Peak MLX memory while the stage runs (GB)")
    ax.set_xlim(0, max(before) * 1.32)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    p = FIG / "stages.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_tradeoff():
    pts = [("all blocks resident", D["res_mlx"], [0.0, 0.0], C["gray"], (-10, 10), "right"),
           ("lag-2 (4 blocks)", D["lag2_mlx"], D["lag2_over"], C["orange"], (10, 8), "left"),
           ("lag-1 (3 blocks), chosen", D["lag1_mlx"], D["lag1_over"], C["aqua"], (-6, -26), "left"),
           ("k=2, no lag (2 blocks)", D["k2_mlx"], D["k2_over"], C["blue"], (10, -6), "left")]
    fig_, ax = plt.subplots(figsize=(6.0, 2.9))
    for name, mem, ov, col, off, ha in pts:
        ax.scatter([mem] * len(ov), ov, s=22, color=col, alpha=0.55, zorder=3)
        m = float(np.median(ov))
        ax.scatter([mem], [m], s=60, color=col, edgecolor=C["ink"], linewidth=0.6, zorder=4)
        ax.annotate(f"{name}\n{mem:.2f} GB, +{m:.1f}%", (mem, m), textcoords="offset points", xytext=off, fontsize=7.2,
                    color=C["ink"], ha=ha)
    ax.axhline(5, color=C["orange"], ls="--", lw=0.9)
    ax.text(8.9, 5.15, "target: +5%", color=C["orange"], fontsize=7, ha="right")
    ax.set_xlabel("Peak MLX memory during denoise (GB)")
    ax.set_ylabel("Extra time per denoise step (%)")
    ax.set_xlim(1.5, 9.5)
    ax.set_ylim(-0.8, 9.5)
    p = FIG / "tradeoff.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_stall():
    st = D["stall"]
    groups = ["warm page cache", "cold SSD (purged)"]
    sync = [np.median(st["sync warm"]), np.median(st["sync cold"])]
    pre = [np.median(st["prefetch warm"]), np.median(st["prefetch cold"])]
    fig_, ax = plt.subplots(figsize=(5.4, 2.5))
    x = np.arange(2)
    ax.bar(x - 0.18, sync, width=0.34, color=C["gray"], label="load, then compute (sync)")
    ax.bar(x + 0.18, pre, width=0.34, color=C["aqua"], label="load next block during compute (prefetch)")
    for i in range(2):
        ax.text(x[i] - 0.18, sync[i] + 0.02, f"{sync[i]:.3f} s", ha="center", va="bottom", fontsize=7.6)
        ax.text(x[i] + 0.18, pre[i] + 0.02, f"{pre[i]:.3f} s", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Time waiting for weights\nper denoise step (s)")
    ax.set_ylim(0, max(sync) * 1.3)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.grid(axis="x", visible=False)
    p = FIG / "stall.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_ceilings():
    rows = [("8-bit weights", D["ceil_q8"], "measured: slower"),
            ("Token merging", D["ceil_merge"], "0.016% real near-duplicates"),
            ("Exact fusion: top-3 candidates", D["ceil_fusion_cands"], "estimate"),
            ("Block-output caching", D["ceil_cache"], "upper bound"),
            ("Exact fusion: ALL non-matmul work removed", D["ceil_fusion_all"], "upper bound")]
    fig_, ax = plt.subplots(figsize=(6.6, 2.7))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, [v - 1 for v in vals], left=1, height=0.55, color=[C["gray"] if v < 1 else C["blue"] for v in vals])
    for i, (n, v, note) in enumerate(rows):
        ax.text(max(v, 1) + 0.006, i, f"{v:.2f}x  ({note})", va="center", fontsize=7.4)
    ax.axvline(1.3, color=C["orange"], ls="--", lw=1)
    ax.text(1.3, len(rows) - 0.35, " goal 1.3x", color=C["orange"], fontsize=7.5, va="bottom")
    ax.axvline(1.0, color=C["ink"], lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.6)
    ax.set_xlim(0.85, 1.36)
    ax.set_xlabel("Best possible end-to-end speed-up (x)")
    ax.grid(axis="y", visible=False)
    p = FIG / "ceilings.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_ops():
    sp = D["op_split"]
    tot = sum(sp.values())
    fig_, ax = plt.subplots(figsize=(6.6, 1.25))
    left = 0
    for (k, lab, col) in (("matmul", "matrix multiplies", C["blue"]), ("attention", "attention", C["aqua"]), ("other", "everything else", C["gray"])):
        w = sp[k] / tot
        ax.barh([0], [w], left=left, color=col, height=0.6)
        ax.text(left + w / 2, 0, f"{lab}\n{sp[k]:.2f} s ({100 * w:.0f}%)", ha="center", va="center", fontsize=7.6,
                color="white" if k != "other" else C["ink"])
        left += w
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    p = FIG / "ops.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def image_grid():
    rows = [("p1", ROOT / "runs/exp-061-full-1/p1.png"), ("p2", ROOT / "runs/exp-062-full-p2/p2.png"), ("p3", ROOT / "runs/exp-062-full-p3/p3.png")]
    rows = [(p, f) for p, f in rows if f.exists()]
    cell, pad, cap = 330, 8, 18
    g = Image.new("RGB", (3 * cell + 4 * pad, len(rows) * (cell + cap + pad) + pad), "white")
    d = ImageDraw.Draw(g)
    try:
        font = ImageFont.truetype(SUP + "Arial.ttf", 13)
    except OSError:
        font = None
    for i, (pid, f) in enumerate(rows):
        a = np.asarray(Image.open(ROOT / "baselines/bf16_free_te" / f"{pid}.png").convert("RGB"))
        b = np.asarray(Image.open(f).convert("RGB"))
        diff = np.clip(np.abs(a.astype(int) - b.astype(int)) * 10, 0, 255).astype(np.uint8)
        y = pad + i * (cell + cap + pad)
        for j, (im, lab) in enumerate(((a, f"{pid}  baseline"), (b, f"{pid}  final 3.5 GB pipeline"), (diff, f"{pid}  |difference| x10"))):
            x = pad + j * (cell + pad)
            g.paste(Image.fromarray(im).resize((cell, cell), Image.LANCZOS), (x, y))
            d.text((x, y + cell + 2), lab, fill=(30, 30, 30), font=font)
    p = FIG / "grid.png"
    g.save(p)
    return p, [r[0] for r in rows]


# ------------------------------------------------------------------ document
def f2(x, n=2):
    return f"{x:.{n}f}"


def build():
    fw, fs, fst, ftr, fc, fo = chart_waterfall(), chart_stall(), chart_stages(), chart_tradeoff(), chart_ceilings(), chart_ops()
    fg, grid_prompts = image_grid()
    psnr = D["final_psnr"]
    worst = min(psnr, key=lambda k: psnr[k]["psnr_db"])
    ps_str = ", ".join(f"{k} {v['psnr_db']:.1f} dB" for k, v in sorted(psnr.items()))
    e2e_lo, e2e_hi = min(D["final_e2e"]), max(D["final_e2e"])
    res_lo, res_hi = min(D["resident_e2e"]), max(D["resident_e2e"])
    lag_lo, lag_hi = min(D["lag1_over"]), max(D["lag1_over"])
    story = []

    # ---------- 1. executive summary
    story += [P("Klein 4B on an Apple M5: running a 16 GB image model in 3.5 GB", "title"),
              P("Project report · FLUX.2 [klein] 4B, text-to-image, 1024×1024 · measured on a MacBook Air M5 (16 GB) · October 2026", "sub"),
              P("Executive summary", "h1"),
              P("Same model, same math, pixel-equivalent (Tier E; weight streaming itself proven bit-identical): peak RAM "
                f"{D['base_fp']:.1f} {ARW} {D['final_fp'][0]:.1f} GB, for ~2–3% extra time. Separately established: Klein's compute is already at "
                f"~{D['matmul_eff_vs_bf16_peak']:.0f}% of this chip's ceiling, so we stopped chasing speed with evidence, not fatigue.", "quote")]
    psnr_rows = " · ".join(f"{k}: {v['psnr_db']:.1f} dB" + (" <b>(worst)</b>" if k == worst else "") for k, v in sorted(psnr.items()))
    story += [table([
        ["", "Baseline", "Final exact pipeline"],
        ["Peak memory (macOS)", f"<b>{D['base_fp']:.2f} GB</b> (exp-011, median of 3; range {min(D['base_fps']):.2f}–{max(D['base_fps']):.2f})",
         f"<b>{D['final_fp'][0]:.2f} GB</b> (exp-061, both rounds)" + (f"; p2/p3 runs: {', '.join(f2(x) for x in D.get('final_fp_p23', []))} GB" if D.get("final_fp_p23") else "")],
        ["End-to-end time per image", f"<b>{D['base_total']:.2f} s</b>: cold single image, stock decode, whole-model compile",
         f"<b>{e2e_lo:.2f}–{e2e_hi:.2f} s</b>: warm, throttled steady state, exact chunked decode. The same pipeline with everything "
         f"resident measured {res_lo:.2f}–{res_hi:.2f} s, so streaming adds no measurable end-to-end time"],
        ["Cost inside the denoise loop", "–", f"+{lag_lo:.1f}–{lag_hi:.1f}% per step (exp-054, two rounds)"],
        ["Quality vs baseline image", "–", f"Streaming is <b>bit-identical</b> to the same-mode resident run ({D['n_stream_runs']}/{D['n_stream_runs']} runs). "
                                           f"End-to-end vs the baseline image: <b>Tier E</b>, {psnr_rows}; worst max pixel diff "
                                           f"{max(v['max_abs_pixel_diff'] for v in psnr.values())}/255, from float rounding in the chunked decode and per-block compile"],
        ["Swap", "flat", "flat in every final-pipeline run"],
    ], [3.6 * cm, 5.2 * cm, 8.2 * cm])]
    story += [Spacer(1, 6),
              P("<b>Why the two times are measured differently.</b> The baseline is a cold single image on a cool chip. The final numbers are "
                f"steady-state runs on a fanless machine that throttles its GPU about {D['throttle_pct']:.0f}% under sustained load, and they include the exact low-memory "
                f"decoder ({min(D['decode_extra_s']):.1f}–{max(D['decode_extra_s']):.1f} s slower than the stock decoder). The like-for-like comparison is the resident run of the same pipeline, "
                "and against that the streaming changes cost nothing measurable end to end."),
              P("<b>What is not claimed yet.</b> We have not run the pipeline on a real 8 GB Mac. 3.5 GB leaves ample room on paper, but "
                "MLX's memory-limit setting turned out to be a soft limit, so no simulated '8 GB' claim is valid. The real-hardware test is pending."),
              P("<b>Speed.</b> The 1.3× speed goal was closed with evidence. The model's matrix multiplies already run at "
                f"{D['matmul_eff_vs_bf16_peak']:.0f}% of the measured peak of the M5's neural accelerators, and every remaining lever has a measured "
                f"ceiling between {D['ceil_q8']:.2f}× and {D['ceil_fusion_all']:.2f}×."),
              PageBreak()]

    # ---------- 2. problem
    story += [P("1. The problem in plain terms", "h1"),
              plain("An image model like Klein works in three stages, a bit like a small print shop. A <b>prompt reader</b> turns your sentence into "
                    "numbers. A <b>factory with 25 workstations</b> then refines a noisy canvas, passing it down the line four times. Finally a "
                    "<b>pixel printer</b> turns the refined canvas into a 1024×1024 picture. Each stage has its own tools, the 'weights', and "
                    "by default the software keeps every tool for every stage on the workbench at once, plus a lot of half-finished material. "
                    "On a 16 GB laptop that does not fit: the computer starts swapping to disk and slows down."),
              precise(f"FLUX.2 [klein] 4B in bf16 has a 7.5 GB Qwen3 text encoder (36 layers, of which 27 are used), a 7.2 GB transformer "
                      f"(5 double-stream + 20 single-stream blocks) and a 0.16 GB VAE. The stock 8-bit mflux setup peaks at "
                      f"{D['waterfall_ref'][1]:.2f} GB and swaps (exp-002). A fully resident bf16 stack exceeds 16 GB and was never runnable."),
              P("The project had two goals of equal weight:", "body"),
              P(f"<b>Speed:</b> at least 1.3× faster end to end with equivalent output. <b>Outcome: closed.</b> The model is already efficient on this chip; section 2 shows the evidence.<br/>"
                f"<b>Memory:</b> peak memory at or below 8 GB (stretch: 6 GB) without worse quality. <b>Outcome: {D['final_fp'][0]:.2f} GB with "
                f"pixel-equivalent output</b>, pending confirmation on real 8 GB hardware."),
              fig(fw, 17.0, f"<b>Figure 1. The memory waterfall.</b> Peak process memory (macOS footprint) as each exact technique is added, bf16 throughout. "
                  f"The hatched bar is the stock 8-bit mflux configuration ({D['waterfall_ref'][2]}), shown for reference. A fully resident bf16 pipeline "
                  "exceeds the machine's 16 GB and was never runnable, so the bf16 chain starts with the text encoder freed after use. "
                  "The first bf16 step still swapped. Every drop is Tier E; the dB range is the chunked decoder's PSNR on the three test prompts."),
              PageBreak()]

    # ---------- 3. speed
    a1, a1c = D["a1"], D["a1c"]
    story += [P("2. The speed investigation (closed)", "h1"),
              plain("Before trying to make the factory faster, we timed every workstation and asked how close each one already is to the "
                    "chip's physical limit. The answer: very close. The heavy lifting is matrix multiplication, and on the M5 it runs on "
                    "dedicated neural-accelerator hardware that is already busy almost all the time. You cannot speed up a machine that "
                    "is already running near full throttle by rearranging the paperwork around it."),
              precise(f"One denoise step takes {D['step_compiled']:.2f} s (compiled, cool chip; exp-030). Microbenchmarks at the exact shapes (exp-031) put "
                      f"the matrix multiplies at {D['op_split']['matmul']:.2f} s and attention at {D['op_split']['attention']:.2f} s per step, "
                      f"leaving {D['op_split']['other']:.2f} s ({100 * D['op_split']['other'] / D['step_compiled']:.0f}%) for everything else. "
                      f"Time-weighted, the matmuls run at {D['matmul_tf_weighted']:.2f} TFLOPS: {D['matmul_eff_vs_bf16_peak']:.0f}% of the measured "
                      f"bf16 peak ({D['peak_bf16_tf']:.2f}) and {D['matmul_eff_vs_any_peak']:.0f}% of the best rate measured in any precision "
                      f"({D['peak_any_tf']:.2f}). Attention runs at {D['sdpa_tf']:.1f} TFLOPS. Forcing a pre-M5 GPU architecture "
                      f"drops peak matmul to {D['peak_bf16_tf_nonax']:.1f} TFLOPS: the baseline already gets a {D['nax_speedup']:.1f}× boost from the neural "
                      "accelerators."),
              fig(fo, 14.0, f"<b>Figure 2. Where one denoise step goes</b> (exp-030 compiled step, split with exp-031 microbenchmarks). "
                  f"Compute-bound work near peak is {100 * (1 - D['op_split']['other'] / D['step_compiled']):.0f}% of the step."),
              P("The closure chain", "h2"),
              P(f"<b>Roofline.</b> {D['matmul_eff_vs_bf16_peak']:.0f}% of peak: the matmuls cannot get meaningfully faster.<br/>"
                f"<b>Exact fusion</b> (merging the small operations between the matmuls). Even if all of the "
                f"{100 * D['op_split']['other'] / D['step_compiled']:.0f}% remainder vanished, the image would be ≤ {D['ceil_fusion_all']:.2f}× faster; "
                f"the three concrete candidates are estimated at ≤ {D['ceil_fusion_cands']:.2f}×.<br/>"
                f"<b>Reusing block outputs between steps (caching).</b> {a1['pct_block_steps_cos_out_gt_0.98']:.1f}% of block-steps changed little "
                f"(cosine > 0.98), but only the first four blocks are stable, about {100 * D['static_frac']:.0f}% of the compute: ceiling ≤ {D['ceil_cache']:.2f}× (exp-040).<br/>"
                f"<b>Merging duplicate tokens.</b> A raw {a1['pct_tokens_near_dup_gt_0.95']:.1f}% of image patches looked like near-copies, but after "
                f"subtracting the average patch only {a1c['pct_tokens_near_dup_gt_0.95_centered']:.3f}% remain: ceiling ≈ {D['ceil_merge']:.2f}× (exp-041).<br/>"
                f"<b>8-bit weights.</b> Measured {D['q8_den']:.2f} s vs {D['bf16_den']:.2f} s for bf16 denoise: {D['ceil_q8']:.2f}×, i.e. slower (exp-015)."),
              plain("The token-merging result needs a word. Picture 4,096 arrows that all lean in roughly the same direction because of a shared "
                    "background component. Measured naively, every arrow looks like a near-copy of its neighbours. Remove the shared lean and "
                    "almost none of them are actually alike. The 'duplication' was a property of the whole crowd, not of individual pairs, "
                    "so there was nothing to merge."),
              fig(fc, 13.0, "<b>Figure 3. Speed ceilings of every lever we examined</b>, as the best possible end-to-end speed-up. None reaches the 1.3× goal "
                  "(dashed). The fusion and caching bars are upper bounds; the 8-bit bar is a measurement."),
              P("We count this as a positive finding: the model is already efficient on this hardware, and we proved it rather than assumed it. "
                "Future speed work belongs on workloads with a different profile: reference-image editing (longer sequences) or "
                "memory-bandwidth-bound models such as LLM decoding, where the deferred fusion pattern is expected to pay."),
              PageBreak()]

    # ---------- 4. memory
    te = D["te"]
    story += [P("3. The memory result", "h1"),
              P("Each stage was taken apart separately. Every technique below keeps the mathematics of the model unchanged. Where results are "
                "bit-identical we say so. Where float rounding changes the last bits we give the PSNR (Tier E means above 40 dB, which is "
                "indistinguishable by eye)."),
              fig(fst, 14.5, "<b>Figure 4. Per-stage memory before and after</b> (MLX peak while the stage runs, same execution mode within each pair)."),
              P("3.1 The pixel printer (VAE decoder)", "h2"),
              plain("The printer seemed to need 9.5 GB, more than the 25-workstation factory. Timing it one step at a time showed that no single step "
                    "needed more than 4.3 GB. The rest was paperwork piling up: the software plans the whole job before doing any of it, and keeps "
                    "every intermediate sheet until the very end. The fix was to print the picture in horizontal strips, carrying a thin overlap "
                    "between strips. One subtlety: the colour-balancing step (GroupNorm) must see the whole page, so we measure the statistics over "
                    "the full page first and only then print strip by strip. That is what makes it exact; the stock tiled mode skips this, which "
                    "leaves a visible seam."),
              precise(f"Stock decode peaks at {D['stage_pairs']['VAE decode'][0]:.2f} GB; the largest single module, profiled eagerly, peaks at "
                      f"{D['vae_profile_max_module']:.2f} GB with bit-identical output (exp-014), so the excess is lazy-graph lifetime. Evaluation barriers "
                      f"between blocks: {D['vae_barrier_mlx']:.2f} GB, bit-identical (exp-018). Exact row-chunked conv / upsample-conv with global-statistic "
                      f"GroupNorm: <b>{D['stage_pairs']['VAE decode'][1]:.2f} GB</b>, PSNR "
                      + ", ".join(f"{k} {v:.1f} dB" for k, v in D["vae_psnr"].items()) + " (exp-019), about 0.8 s slower."),
              P("3.2 The 25-workstation factory (transformer)", "h2"),
              plain("Instead of keeping all 25 workstations' tools on the floor, we keep only the station that is working and bring in the next one's tools "
                    "while it works, then clear the previous station. The obvious worry was that fetching tools from the disk would make the "
                    f"line wait. It doesn't: fetching one station's tools takes about {100 * D['load_compute_ratio']:.0f}% of the time that station spends working, so the "
                    "delivery is always early. The real cost turned out to be different. Forcing the line to stop and check after every station "
                    "left small gaps. Letting one station's work stay queued while the next begins (a 'lag of one') closed most of them."),
              precise(f"Each block's weights live in their own file ({D['block_gb']:.3f} GB on average). Cold SSD read speed after a page-cache purge: "
                      f"{min(D['ssd_cold']):.2f}–{max(D['ssd_cold']):.2f} GB/s, so about {D['block_load_ms']:.0f} ms to load a block vs about "
                      f"{D['block_compute_ms']:.0f} ms to compute it. Prefetching the next block hides <b>{D['hidden_pct_cold']:.1f}%</b> of the cold load stall "
                      f"(exp-052). Eager streaming still cost +{D['prefetch_cold_over']:.1f}% per step, and +{D['barrier_over']:.1f}% of that came from "
                      f"the per-block synchronisation alone. With each block compiled separately and a lag-1 barrier, the cost is "
                      f"<b>+{lag_lo:.1f}–{lag_hi:.1f}% per step</b> and the denoise peak is <b>{D['lag1_mlx']:.2f} GB</b> instead of {D['res_mlx']:.2f} GB "
                      f"(exp-054). The output is bit-identical to the resident run of the same mode in all {D['n_stream_runs']} streaming runs."),
              fig(fs, 11.5, f"<b>Figure 5. Waiting time per denoise step.</b> Loading the next block while the current one computes hides "
                  f"{D['hidden_pct_cold']:.1f}% of the cold-SSD stall (exp-051 warm, exp-052 cold; medians of two rounds)."),
              fig(ftr, 12.5, "<b>Figure 6. The residency knob.</b> More blocks in flight cost memory and buy back the synchronisation gaps. "
                  "Lag-1 meets the +5% target; lag-2 buys nothing more (exp-053, exp-054; small dots are individual rounds)."),
              P("3.3 The prompt reader (text encoder)", "h2"),
              plain("The prompt reader is a 36-layer language model, but FLUX.2 only reads its notes from layers 9, 18 and 27. The last nine layers "
                    "do work nobody uses. We also noticed it was loading a 0.7 GB dictionary of 151,936 words to look up at most 512 of them. "
                    "So we now read only the 27 useful layers, one at a time, and look up only the words in the prompt."),
              precise(f"Stock peak {te['stock']['peak_mlx_first_gb']:.2f} GB, of which the used weights are "
                      f"{te['stock']['anatomy']['used_weights_gb']:.2f} GB; the lazy-graph share is small. Per-layer evaluation barriers made it "
                      f"<b>worse</b>, {te['barrier']['peak_mlx_first_gb']:.2f} GB, because forcing evaluation also runs the nine dead layers that "
                      f"the lazy graph had skipped. Streaming the 27 used layers plus a row-gathered embedding: "
                      f"<b>{te['stream']['peak_mlx_first_gb']:.2f} GB</b>, bit-identical, and the cold encode got faster "
                      f"({te['stream']['encode_s_first']:.2f} s vs {te['stock']['encode_s_first']:.2f} s; exp-060)."),
              P("3.4 The composed pipeline", "h2"),
              precise(f"All three together (exp-061, cold SSD, two rounds): <b>{D['final_fp'][0]:.2f} GB</b> peak process memory, swap flat, image "
                      "bit-identical to the same pipeline with a resident text encoder. The peak is now the denoise stage "
                      f"({D['final_mlx'][0]:.2f} GB of MLX arrays plus the Python and MLX runtime). Stage peaks: "
                      + ", ".join(f"{k} {v:.2f} GB" for k, v in D["final_stage_fp"].items() if not k.startswith("denoise_run")) + "."),
              fig(fg, 15.5, f"<b>Figure 7. Baseline vs the final 3.5 GB pipeline</b>, prompts {', '.join(grid_prompts)}. PSNR vs baseline: {ps_str} "
                  f"(worst: {worst}). The difference images are amplified ten times; they show float-rounding noise, not structure."),
              PageBreak()]

    # ---------- 5. honesty
    eg = D["eager_vs_compiled"]
    story += [P("4. How we kept it honest", "h1"),
              P("A memory or speed claim is only as good as the measurement behind it. These are the rules we followed and the traps they caught."),
              P("A fanless laptop gets slower as it warms up", "h2"),
              plain("The test machine has no fan. After a few minutes of work it gets warm and quietly slows down, so the fifth run of anything is "
                    "slower than the first. Comparing 'old method first, new method second' would therefore flatter whichever ran first."),
              precise(f"Five back-to-back runs drifted from {D['thermal']['exp-001'][0]:.1f} s to {D['thermal']['exp-001'][1]:.1f} s (exp-001) and "
                      f"{D['thermal']['exp-002'][0]:.1f} s to {D['thermal']['exp-002'][1]:.1f} s (exp-002). Sustained runs hold the GPU at "
                      f"~{D['throttle_mhz'][0]} MHz vs ~{D['throttle_mhz'][1]} MHz cold, while macOS still reports thermal pressure 'Nominal'. "
                      "Every comparison therefore alternates A and B in separate processes with a 150 s cool-down gated on thermal state, and "
                      "powermetrics logs the GPU clock for every run."),
              P("Warm vs cold storage", "h2"),
              plain(f"The first streaming test looked too good: weights 'loaded' at {D['warm_read_gbps']:.0f} GB/s, faster than the SSD can read. The 16 GB machine was "
                    "silently keeping the files in spare memory. An 8 GB Mac could not do that, so we emptied that cache before every step to force "
                    "real disk reads."),
              precise(f"Warm page-cache reads: {min(D['ssd_warm']):.1f}–{max(D['ssd_warm']):.1f} GB/s. After `purge`: "
                      f"{min(D['ssd_cold']):.2f}–{max(D['ssd_cold']):.2f} GB/s. All headline streaming numbers are cold."),
              P("Predict first, then measure", "h2"),
              precise(f"Before the cold test, this prediction was written down: a block is ~0.29 GB; at ≥ 2 GB/s it loads in < 150 ms vs ~180 ms of compute, "
                      f"so prefetch should hide cold reads with margin. Measured: {D['block_gb']:.3f} GB, {min(D['ssd_cold']):.1f} GB/s, "
                      f"{D['block_load_ms']:.0f} ms vs {D['block_compute_ms']:.0f} ms, {D['hidden_pct_cold']:.1f}% hidden. <b>Confirmed.</b> "
                      "Writing the expectation down first keeps a lucky number from being read as a designed one."),
              P("Compare like with like (rule R1)", "h2"),
              plain("The same model run two slightly different ways (all at once, or step by step) can produce images that differ in the last "
                    "decimal places. If we compared a step-by-step candidate against an all-at-once reference, we could blame our change for a "
                    "difference it did not cause."),
              precise("Eager vs compiled latents after four steps: cosine " + ", ".join(f"{k} {v['cos']:.4f}" for k, v in eg.items())
                      + " (exp-040). Every bit-identity claim is made against a reference produced in the same execution mode."),
              P("Swapping runs don't count (R3) and simulations aren't evidence (R2)", "h2"),
              precise(f"Any run whose swap usage grew is marked 'memory-only', and its time is never quoted. MLX's memory limit is soft: capped at 8 GB, a run "
                      f"still peaked at {D['cap_runs'][8][1]:.2f} GB; capped at 6 GB, at {D['cap_runs'][6][1]:.2f} GB (exp-022). So no '8 GB machine' "
                      "claim is made from a simulation."),
              P("Two quality tiers", "h2"),
              plain("Some changes only reorder arithmetic, so the picture is the same to within rounding: <b>Tier E</b>, checked by PSNR above 40 dB "
                    "or bit-identity. Others genuinely approximate, like smaller number formats: <b>Tier A</b>, checked with a perceptual metric "
                    "(LPIPS below 0.05) and side-by-side images. Nothing in the headline is Tier A."),
              PageBreak()]

    # ---------- 6. what didn't work
    lq, lt = D["lpips_q8"]["per_prompt"], D["lpips_tiled"]["per_prompt"]
    story += [P("5. What didn't work", "h1"),
              table([["Attempt", "Number that ended it"],
                     ["8-bit weights for speed", f"denoise {D['q8_den']:.2f} s vs {D['bf16_den']:.2f} s bf16: {100 * (D['q8_den'] / D['bf16_den'] - 1):.0f}% slower (compute-bound; unpacking adds work)"],
                     ["8-bit weights for memory (Tier A)", "LPIPS vs bf16 " + " / ".join(f"{v['lpips']:.3f}" for v in lq.values()) + ": fails 0.05 on p3"],
                     ["mflux tiled VAE", "LPIPS " + " / ".join(f"{v['lpips']:.3f}" for v in lt.values()) + " vs bf16 and a visible seam (per-tile GroupNorm)"],
                     ["Per-layer barriers on the text encoder", f"{te['barrier']['peak_mlx_first_gb']:.2f} GB vs {te['stock']['peak_mlx_first_gb']:.2f} GB stock: forced the dead layers to run"],
                     ["mx.set_memory_limit as an 8 GB simulation", f"soft limit: {D['cap_runs'][8][1]:.2f} GB at an 8 GB cap, {D['cap_runs'][6][1]:.2f} GB at 6 GB"],
                     ["Block-output caching", f"only ~{100 * D['static_frac']:.0f}% of compute is static: ≤ {D['ceil_cache']:.2f}×"],
                     ["Token merging", f"{a1c['pct_tokens_near_dup_gt_0.95_centered']:.3f}% near-duplicates after de-meaning"],
                     ["Exact kernel fusion (deferred)", f"remainder is {100 * D['op_split']['other'] / D['step_compiled']:.0f}% of a step: ≤ {D['ceil_fusion_all']:.2f}×"],
                     ["Lag-2 streaming", f"+{np.median(D['lag2_over']):.1f}% vs +{np.median(D['lag1_over']):.1f}% for lag-1, with 0.5 GB more resident"]],
                    [5.2 * cm, 11.8 * cm])]

    # ---------- 7. patterns
    story += [P("6. Reusable patterns", "h1"),
              P("The point of the project is an optimizer whose lessons carry to other models. Each entry below is written up in "
                "<font name='Sans'>library/patterns/</font> with no reference to Klein in its matching rule."),
              table([["Pattern", "Applies beyond Klein to", "Measured here", "Tier"],
                     ["Exact row-chunked VAE decode (global GroupNorm stats)", "every convolutional VAE: SD, SDXL, Flux, video VAEs",
                      f"{D['stage_pairs']['VAE decode'][0]:.2f} {ARW} {D['stage_pairs']['VAE decode'][1]:.2f} GB", "E"],
                     ["Eval barriers between blocks", "any lazily evaluated graph (MLX, JAX)", f"VAE {D['stage_pairs']['VAE decode'][0]:.2f} {ARW} {D['vae_barrier_mlx']:.2f} GB; never before truncating dead layers", "E, identical"],
                     ["Block weight streaming + prefetch + lag-1", "DiTs, LLM layers, any sequential block stack",
                      f"{D['res_mlx']:.2f} {ARW} {D['lag1_mlx']:.2f} GB, +{lag_lo:.1f}–{lag_hi:.1f}%/step", "E, identical"],
                     ["Dead-layer truncation + row-gathered embeddings", "any encoder used for intermediate hidden states; any large vocabulary",
                      f"{te['stock']['peak_mlx_first_gb']:.2f} {ARW} {te['stream']['peak_mlx_first_gb']:.2f} GB", "E, identical"],
                     ["Allocator cache cap", "any MLX workload; PyTorch caching allocator analogue", f"macOS footprint −{D['cachecap_saving']:.2f} GB, no speed cost", "E, identical"],
                     ["Lifecycle: free each stage after last use", "any multi-model pipeline", f"{D['lifecycle_mlx'][0]:.2f} {ARW} {D['lifecycle_mlx'][1]:.2f} GB MLX (8-bit stack)", "E, identical"],
                     ["Per-layer quantization sensitivity scan", "any model (tool, not rewrite)", f"{len(D['sens']['ranking_q4'])} layers ranked; floor verified 0", "tool"]],
                    [4.6 * cm, 4.6 * cm, 5.4 * cm, 2.4 * cm]),
              Spacer(1, 8),
              P("<b>The residency knob</b> (transformer denoise, cold SSD):"),
              table([["Setting", "Blocks in memory", "Denoise peak (MLX)", "Extra time per step"],
                     ["k=2, no lag", "2", f"{D['k2_mlx']:.2f} GB", f"+{min(D['k2_over']):.1f}–{max(D['k2_over']):.1f}%"],
                     ["lag-1 (chosen)", "3", f"{D['lag1_mlx']:.2f} GB", f"+{lag_lo:.1f}–{lag_hi:.1f}%"],
                     ["lag-2", "4", f"{D['lag2_mlx']:.2f} GB", f"+{min(D['lag2_over']):.1f}–{max(D['lag2_over']):.1f}%"],
                     ["all resident", "25", f"{D['res_mlx']:.2f} GB", "0 (reference)"]],
                    [4.0 * cm, 3.2 * cm, 4.2 * cm, 5.6 * cm]),
              Spacer(1, 6),
              P(f"<b>Lesson: streaming cost is bubbles, not loading.</b> With prefetch, waiting for weights fell to "
                f"{np.median(D['stall']['prefetch cold']) * 1000:.1f} ms per step, yet streaming still cost +{D['prefetch_cold_over']:.1f}% (eager). "
                f"Synchronisation gaps explained +{D['barrier_over']:.1f}%, and the remaining {D['contention_pct'][0]:.1f}–{D['contention_pct'][1]:.1f}% is memory-bus contention between the weight transfer "
                "and the GPU, now recorded in the hardware profile as a generic cost of streaming on unified memory."),
              PageBreak()]

    # ---------- 8. open items
    story += [P("7. Open items", "h1"),
              P("• <b>Real 8 GB Mac test.</b> The only valid confirmation of the memory result. The candidate local machine (M1, 8 GB) "
                "needs about 20 GB of free disk first; an M1 also lacks the M5's neural accelerators, so it would confirm memory, not speed.<br/>"
                "• <b>Generalization (Phase 4).</b> Run the frozen patterns on Z-Image Turbo and a small LLM decoder, and record which patterns fire.<br/>"
                "• <b>Upstreaming.</b> The exact chunked VAE decode, text-encoder dead-layer truncation and block streaming are candidates for mflux.<br/>"
                "• <b>Speed on other workloads.</b> Reference-image edits and bandwidth-bound LLM decoding, where the deferred fusion pattern applies."),
              PageBreak()]

    # ---------- 9. appendix
    rows = experiment_rows(D["base_total"])
    story += [P("Appendix A. Every experiment", "h1"),
              P(f"Speed-up is against the {D['base_total']:.2f} s baseline (exp-011, median of 3) and is shown only for end-to-end runs with flat swap. "
                "'memory-only' means swap grew, so the time is invalid (R3). PSNR is against the baseline images; 'same-mode' is the R1 bit-identity "
                "check. Profiling, microbenchmark and stage-only runs have no end-to-end time.", "small"),
              table([["Exp", "Config", "Cache cap GB", "Quality ref", "Total s", "Speed-up", "OS peak GB", "PSNR dB / same-mode", "Swap"]] + rows,
                    [2.5 * cm, 3.6 * cm, 1.1 * cm, 2.0 * cm, 1.2 * cm, 1.6 * cm, 1.3 * cm, 3.4 * cm, 0.9 * cm], size="tiny"),
              PageBreak(),
              P("Appendix B. Machine, software, protocol", "h1"),
              table([["", ""],
                     ["Machine", "MacBook Air Mac17,3 (fanless), Apple M5: 4 performance + 6 efficiency CPU cores, 8-core GPU with neural accelerators, Metal 4, 16 GB unified memory, macOS 26.6"],
                     ["Measured device facts", f"bf16 matmul peak {D['peak_bf16_tf']:.2f} TFLOPS ({D['peak_bf16_tf_nonax']:.2f} without neural accelerators); "
                                               f"memory bandwidth {D['bandwidth']:.0f} GB/s (read+write); SSD {min(D['ssd_cold']):.2f}–{max(D['ssd_cold']):.2f} GB/s cold"],
                     ["Software", "Python 3.12.15, MLX 0.32.4, mflux 0.23.0, numpy 2.5.4, LPIPS (AlexNet) via torch 2.14.1"],
                     ["Model", "black-forest-labs/FLUX.2-klein-4B @ e7b7dc2, Apache-2.0; bf16; 1024×1024, 4 steps, guidance 1.0, seed 42; prompts p1 portrait, p2 landscape, p3 text sign"],
                     ["R1 execution mode", "every correctness comparison uses a reference produced in the same execution mode (eager vs compiled)"],
                     ["R2 simulation", "mx.set_memory_limit is a soft limit; no smaller-machine claim is made from it"],
                     ["R3 timing validity", "swap-grown runs are memory-only; timing comparisons alternate A/B processes with cool-downs"],
                     ["Tiers", "E: PSNR > 40 dB or bit-identical. A: LPIPS < 0.05 plus visual review"]],
                    [3.4 * cm, 13.6 * cm], header=False)]

    def on_page(c, doc):
        c.saveState()
        c.setFont("Sans", 7.5)
        c.setFillColor(MUTE)
        c.drawString(2 * cm, 1.2 * cm, "Klein 4B on Apple M5 · kernelforge")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        c.restoreState()

    out = HERE / "klein-m5-report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Klein 4B on an Apple M5: running a 16 GB image model in 3.5 GB", author="kernelforge")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(out)


if __name__ == "__main__":
    build()
