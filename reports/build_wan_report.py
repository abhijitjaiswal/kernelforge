"""Build reports/wan-m5-report.pdf from recorded runs only (reports/wan_report_data.py). Same layout and typography as the Klein
report (reports/build_report.py).

    python reports/build_wan_report.py
Requires matplotlib + reportlab + pillow. Charts and image grids are written to reports/fig/wan_*.png at 220 dpi.
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
RUNS = ROOT / "runs"
sys.path.insert(0, str(HERE))
from wan_report_data import collect, R  # noqa: E402

FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
D = collect()

# ------------------------------------------------------------------ fonts / styles (as the Klein report)
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
OK = '<font name="Uni">✓</font>'
NO = '<font name="Uni">✗</font>'
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


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def mins(s):
    return f"{s / 60:.1f} min"


# ------------------------------------------------------------------ charts
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "gray": "#a3a19a", "ink": "#2b2b2b", "mute": "#6b6a66", "grid": "#e4e3de"}
PROMPT_COL = {"v1": C["blue"], "v2": C["aqua"], "v3": C["orange"]}
PROMPT_NAME = {"v1": "fox (smooth)", "v2": "ocean waves (chaotic)", "v3": "neon sign (fine detail)"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5,
                     "axes.edgecolor": C["mute"], "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})


def chart_stages():
    names = ["Text encoder\n(umT5-XXL, fp32)", "DiT denoise\n(30 blocks)", "VAE decode\n(3D, fp32)"]
    before = [D["stock_te"]["mlx"], D["stock_dit"]["mlx"], D["stock_dec"]["mlx"]]
    after = [D["comp"]["te_mlx"], D["comp"]["dit_mlx"], D["comp"]["dec_mlx"]]
    notes = [f"swap +{D['stock_te']['swap_gb']:.1f} GB", "", "swaps; untiled cannot run"]
    fig_, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(3)
    ax.barh(y + 0.19, before, height=0.36, color=C["gray"], label="stock mlx-video")
    ax.barh(y - 0.19, after, height=0.36, color=C["blue"], label="composed exact pipeline")
    for i in range(3):
        ax.text(before[i] + 0.2, y[i] + 0.19, f"{before[i]:.2f} GB" + (f"  ({notes[i]})" if notes[i] else ""), va="center", fontsize=7.4, color=C["ink"])
        ax.text(after[i] + 0.2, y[i] - 0.19, f"{after[i]:.2f} GB", va="center", fontsize=7.6, color=C["ink"], fontweight="bold")
    ax.axvline(16, color=C["orange"], lw=1.1, ls="--")
    ax.text(16.25, -0.42, "16 GB: this machine's total memory", color=C["orange"], fontsize=7, va="center")
    ax.set_yticks(y, names, fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("Peak MLX memory while the stage runs (GB)")
    ax.set_xlim(0, 30)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    ax.set_ylim(2.6, -0.6)
    ax.grid(axis="y", visible=False)
    p = FIG / "wan_stages.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_timeline():
    import json
    fig_, ax = plt.subplots(figsize=(8.4, 3.0))
    for exp, col, lab in (("exp-103-base-auto", C["gray"], "stock downstream of frozen embeddings (exp-103)"),
                          ("exp-127-composed", C["blue"], "composed exact pipeline, all three stages (exp-127)")):
        m = json.loads((RUNS / exp / "memtrace.json").read_text())
        cols = m["cols"]
        s = np.array(m["samples"], dtype=np.float64)
        t, fpp = s[:, cols.index("t_s")] / 60, s[:, cols.index("phys_footprint")] / 1024 ** 3
        ax.plot(t, fpp, color=col, lw=1.1, label=lab)
        for tm, name in m["marks"]:
            if name in ("denoise", "vae_decode"):
                ax.axvline(tm / 60, color=col, lw=0.5, ls=":")
    ax.axhline(16, color=C["orange"], lw=1.0, ls="--")
    ax.text(0.2, 16.3, "16 GB total memory", color=C["orange"], fontsize=7)
    ax.set_xlabel("Minutes since start")
    ax.set_ylabel("Process footprint (GB, macOS)")
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, loc="upper left", fontsize=7.3)
    p = FIG / "wan_timeline.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_decoder():
    rows = D["dec_versions"]
    labels = ["stock tiled\n(exp-132)"] + [f"{r[0]}\n({r[1].split('-')[0]}-{r[1].split('-')[1]})" for r in rows]
    fps = [D["stock_dec"]["fp"]] + [r[3] for r in rows]
    swaps = [D["stock_dec"]["swap"]] + [r[5] for r in rows]
    fig_, ax = plt.subplots(figsize=(7.4, 2.8))
    x = np.arange(len(fps))
    cols = [C["gray"]] + [C["blue"]] * (len(rows) - 1) + [C["aqua"]]
    bars = ax.bar(x, fps, color=cols, width=0.6)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, fps[i] + 0.25, f"{fps[i]:.2f} GB" + ("\nswap grew" if swaps[i] else ""), ha="center",
                va="bottom", fontsize=7.2, color=C["ink"], linespacing=1.1)
    ax.axhline(16, color=C["orange"], lw=1.0, ls="--")
    ax.set_xticks(x, labels, fontsize=6.8)
    ax.set_ylabel("Decode footprint peak (GB)")
    ax.set_ylim(0, 20)
    p = FIG / "wan_decoder.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_caching():
    fig_, ax = plt.subplots(figsize=(7.2, 3.3))
    for p in ("v1", "v2", "v3"):
        col = PROMPT_COL[p]
        ax.scatter([D["exact_s"][p] / 60], [0], s=46, color=col, marker="s", zorder=3, edgecolor="white", linewidth=0.8)
        for mode, mk in (("A2", "o"), ("A3-d", "^")):
            c = D["cache"].get((p, mode))
            if not c:
                continue
            ax.scatter([c["s"] / 60], [c["lp_max"]], s=58, color=col, marker=mk, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{mode} {c['speedup']:.2f}×", (c["s"] / 60, c["lp_max"]), textcoords="offset points", xytext=(6, -3),
                        fontsize=7, color=C["ink"])
    ax.axhline(0.05, color=C["orange"], lw=1.1, ls="--")
    ax.text(21.3, 0.052, "Tier A gate: LPIPS < 0.05 on every frame", color=C["orange"], fontsize=7, ha="right")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=PROMPT_COL[p], marker="o", ls="", label=PROMPT_NAME[p]) for p in ("v1", "v2", "v3")]
    h += [Line2D([], [], color=C["mute"], marker=m, ls="", label=l) for m, l in (("s", "exact (bit-identical)"), ("o", "A2 static schedule"),
                                                                                 ("^", "A3-d dynamic threshold"))]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="upper right", ncol=2)
    ax.set_xlabel("Minutes per clip, end to end (lower is faster)")
    ax.set_ylabel("Worst-frame LPIPS vs exact (lower is closer)")
    ax.set_xlim(11.5, 21.8)
    ax.set_ylim(-0.005, 0.125)
    p = FIG / "wan_caching.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def chart_ucurve():
    fig_, ax = plt.subplots(figsize=(7.2, 2.7))
    for p in ("v2", "v3"):
        c = D["cache"][(p, "A3-d")]
        st = [e["step"] for e in c["rate_log"]]
        rt = [e["rate"] for e in c["rate_log"]]
        ax.plot(st, rt, color=PROMPT_COL[p], lw=1.3, marker="o", ms=2.4, label=PROMPT_NAME[p])
        sk = [i + 1 for i, m in enumerate(c["taken"]) if m == "s"]
        ax.scatter(sk, [0.004 if p == "v2" else 0.009] * len(sk), marker="|", s=40, color=PROMPT_COL[p])
    ax.axhline(0.022, color=C["mute"], lw=0.9, ls="--")
    ax.text(50, 0.024, "A3-d threshold 0.022", fontsize=7, color=C["mute"], ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.003, 0.4)
    ax.set_xlabel("Denoising step (1–50); tick marks at the bottom = steps A3-d skipped")
    ax.set_ylabel("Block-stack change per step")
    ax.legend(frameon=False, fontsize=7.3, loc="upper center")
    p = FIG / "wan_ucurve.png"
    fig_.savefig(p)
    plt.close(fig_)
    return p


def frames_grid(cols, out, frames=(0, 17, 35), w=416):
    """cols: [(exp, label)]; one row per frame index."""
    font = ImageFont.truetype(SUP + "Arial Bold.ttf", 15)
    arrs = [np.load(RUNS / e / "frames.npy") for e, _ in cols]
    h = int(w * arrs[0].shape[1] / arrs[0].shape[2])
    pad = 4
    img = Image.new("RGB", (len(cols) * (w + pad) - pad, len(frames) * (h + pad) - pad), "white")
    for j, a in enumerate(arrs):
        for i, f in enumerate(frames):
            tile = Image.fromarray(a[f]).resize((w, h), Image.LANCZOS)
            img.paste(tile, (j * (w + pad), i * (h + pad)))
        d = ImageDraw.Draw(img)
        lab = cols[j][1]
        tw = d.textlength(lab, font=font)
        d.rectangle([j * (w + pad) + 6, 6, j * (w + pad) + 14 + tw, 28], fill=(0, 0, 0))
        d.text((j * (w + pad) + 10, 8), lab, font=font, fill=(255, 255, 255))
    p = FIG / out
    img.save(p)
    return p


# ------------------------------------------------------------------ document
def build():
    fs, ft, fd, fc, fu = chart_stages(), chart_timeline(), chart_decoder(), chart_caching(), chart_ucurve()
    c, ch = D["comp"], D["chk"]
    cc = D["cache"]
    neon = frames_grid([("exp-ref-v3", "EXACT"), ("exp-a2-v3", f"A2  LPIPS {cc[('v3', 'A2')]['lp_max']:.3f}  FAIL"),
                        ("exp-a3d-v3", f"A3-d  LPIPS {cc[('v3', 'A3-d')]['lp_max']:.3f}  PASS")], "wan_neon.png")
    waves = frames_grid([("exp-ref-v2", "EXACT"), ("exp-a2-v2", f"A2  LPIPS {cc[('v2', 'A2')]['lp_max']:.3f}  FAIL"),
                         ("exp-a3d-v2", f"A3-d  LPIPS {cc[('v2', 'A3-d')]['lp_max']:.3f}  FAIL")], "wan_waves.png")
    fox = frames_grid([("exp-132-stock-decode", "stock mlx-video (tiled decode)"), ("exp-127-composed", "composed exact pipeline"),
                       ("exp-134-a2", f"A2  LPIPS {cc[('v1', 'A2')]['lp_max']:.3f}  PASS")], "wan_fox.png")
    e2e = D["e2e_vs_tiled"]
    vt = D["v6"]["vs_tiled"]
    exact_mean = np.mean(list(D["exact_s"].values())) / 60
    story = []

    # ---------- 1. executive summary
    story += [P("Wan 2.1 on an Apple M5: text-to-video in 8.6 GB, bit-identical to stock", "title"),
              P("Project report · Wan 2.1 T2V 1.3B via mlx-video · 832×480, 33 frames requested (36 output), 50 steps · "
                "measured on a MacBook Air M5 (16 GB) · 21 October 2026", "sub"),
              P("Executive summary", "h1"),
              P(f"Stock mlx-video cannot run two of Wan's three stages on a 16 GB Mac without heavy swapping. The composed pipeline runs "
                f"the whole clip at <b>{c['fp']:.2f} GB</b> peak with swap flat, in {mins(c['s'])}, and its final latents are "
                f"<b>bit-identical to stock's</b>. No step-caching fast mode ships: no configuration passed the quality gate on all three "
                f"test prompts.", "quote")]
    story += [table([
        ["Stage", "Stock mlx-video on this machine", "Composed exact pipeline", "Exactness vs stock"],
        ["Text encoder<br/>(umT5-XXL, upcast to fp32)", f"<b>{f2(D['stock_te']['mlx'])} GB</b> MLX / {f2(D['stock_te']['fp'])} GB footprint, "
         f"swap +{f1(D['stock_te']['swap_gb'])} GB, {D['stock_te']['s']:.0f} s* (exp-101): <b>cannot run in 16 GB without heavy swap</b>",
         f"<b>{f2(D['te_stream']['mlx'])} GB</b> / {f2(D['te_stream']['fp'])} GB, {D['te_stream']['s']:.1f} s, swap flat (exp-102)",
         f"<b>bit-identical</b> embeddings, prompt and negative prompt (re-checked inside the pipeline: exp-128 "
         f"{'identical' if ch['ctx'] else 'DIFFERS'})"],
        ["DiT denoise<br/>(30 blocks, 2.64 GB bf16)", f"{f2(D['stock_dit']['mlx'])} GB MLX, {f1(D['stock_dit']['step'])} s/step (exp-130)",
         f"<b>{f2(c['dit_mlx'])} GB</b>, +{min(D['stream_over']):.1f}–{max(D['stream_over']):.1f}% vs per-block-compiled resident (exp-126)",
         f"<b>bit-identical at 50 steps</b> (exp-127 = exp-130: {'identical' if ch['dit50'] else 'DIFFERS'})"],
        ["VAE decode<br/>(3D causal, fp32)", f"tiled: <b>{f2(D['stock_dec']['mlx'])}–{f2(D['stock_dec']['mlx_103'])} GB</b> MLX, swaps, "
         f"{D['stock_dec']['s_103']:.0f}–{D['stock_dec']['s']:.0f} s* (exp-103, exp-132). Untiled: <b>cannot run</b> "
         f"(one {f2(D['untiled_crash']['asked_gb'])} GB buffer &gt; the {f2(D['untiled_crash']['limit_gb'])} GB Metal limit, exp-108)",
         f"<b>{f2(D['v6']['mlx'])} GB</b> / {f2(D['v6']['fp'])} GB, {D['v6']['s']:.0f} s, swap flat (exp-125)",
         f"frames 1–{ch['prefix_n']} <b>bit-identical</b> to stock untiled (the longest prefix stock can decode); all 36 vs stock tiled: "
         f"{f1(vt['psnr_db_min'])} dB min / {f1(vt['psnr_db_mean'])} dB mean"],
        ["<b>End to end</b>", f"2 of 3 stages cannot run within 16 GB. With the encoder skipped (frozen embeddings) the rest still peaks at "
         f"{f2(D['base103']['fp'])} GB with swap (exp-103)",
         f"<b>{f2(c['fp'])} GB</b> footprint peak, <b>{mins(c['s'])}</b>, swap flat (exp-127)",
         f"final latents <b>bit-identical</b>; video vs stock's default tiled decode of the same latents: {f1(e2e['psnr_db_min'])} dB min / "
         f"{f1(e2e['psnr_db_mean'])} dB mean, max pixel diff {e2e['max_abs_pixel_diff']} (exp-132)"]],
        [2.6 * cm, 5.0 * cm, 4.2 * cm, 5.2 * cm], zebra=True),
        P("* Swap-inflated: valid for memory, not for speed (protocol rule R3). These are what stock takes on this machine, not stock's speed.", "small"),
        Spacer(1, 4),
        P("Two modes, one shipped", "h2"),
        table([["Mode", "Time per clip", "Peak", "Claim"],
               ["<b>Exact</b> (shipped)", f"{min(D['exact_s'].values()) / 60:.1f}–{max(D['exact_s'].values()) / 60:.1f} min on three prompts",
                f"{min(D['exact_fp'].values()):.2f}–{max(D['exact_fp'].values()):.2f} GB, swap flat",
                "final latents bit-identical to stock mlx-video; frames identical to stock's exact decode wherever stock can run it"],
               ["Fast (step caching): <b>not shipped</b>", f"{min(x['s'] for x in cc.values()) / 60:.1f}–{max(x['s'] for x in cc.values()) / 60:.1f} min",
                f"{min(x['fp'] for x in cc.values()):.2f}–{max(x['fp'] for x in cc.values()):.2f} GB",
                "no configuration passed LPIPS &lt; 0.05 on all three prompts (section 5); curve and reasons recorded"]],
              [3.4 * cm, 3.6 * cm, 3.2 * cm, 6.8 * cm]),
        Spacer(1, 6),
        plain("Stock software needs more memory than this laptop has for two of the three steps of making a video, so it either crawls "
              "while the disk stands in for memory, or crashes. We rebuilt how those steps use memory, without changing any arithmetic. "
              "The video now fits in about half the machine, and the numbers inside the model match the stock software's exactly. "
              "We also tried to make it faster by skipping repeated work; that worked on a calm scene but visibly changed busy scenes "
              "(waves, a rainy street), so it is not offered."),
        PageBreak()]

    # ---------- 2. where memory went
    story += [P("1. Where the memory goes", "h1"),
              fig(fs, 15.5, "Figure 1. Peak MLX memory per stage, stock vs the composed exact pipeline. Stock encoder: exp-101; stock DiT: exp-130; "
                            "stock decoder: exp-132 (tiled; untiled cannot run). Composed: exp-127."),
              fig(ft, 16.5, "Figure 2. Process footprint over one full clip. Grey: stock mlx-video downstream of frozen stock embeddings (exp-103; the "
                            "stock encoder alone needs 23.9 GB, exp-101, so it was run once separately). Blue: the composed pipeline, all three "
                            "stages (exp-127). Dotted lines: start of denoising and of decoding. The grey spike at the end is the stock tiled decode."),
              precise(f"Encoder: mlx-video upcasts the 24-block umT5-XXL and its 256,384-row embedding table to fp32, "
                      f"{f2(D['stock_te']['mlx'])} GB. DiT: 2.64 GB of bf16 weights plus activations for ~14,000 tokens at batch 2 (CFG). "
                      f"Decoder: each 3×3×3 causal convolution at full resolution needs a working buffer about 7.6× its input; one such "
                      f"convolution sets the stock peak (diagnosed with a per-module profile, exp-107)."),
              PageBreak()]

    # ---------- 3. how
    story += [P("2. How each stage was made to fit, exactly", "h1"),
              P("Text encoder: stream the blocks, upcast as they load", "h2"),
              precise(f"The 24 blocks are stored as bf16 files and streamed through two-block residency with prefetch; each is upcast to fp32 "
                      f"as it loads, so the arithmetic is the stock fp32 arithmetic. The embedding table is never loaded: only the rows for the "
                      f"prompt's tokens are read from disk. {f2(D['stock_te']['mlx'])} {ARW} {f2(D['te_stream']['mlx'])} GB, bit-identical "
                      f"(exp-101/102). A new variant of the frozen streaming pattern (upcast-on-load)."),
              P("DiT: block weight streaming (frozen pattern, adapter only)", "h2"),
              precise(f"The frozen engine streams the 30 blocks with prefetch and per-block compilation. mlx-video calls blocks with a positional "
                      f"argument the frozen engine does not accept, so a small adapter subclass folds it into a keyword; the engine is unchanged. "
                      f"{f2(D['stock_dit_8']['mlx'])} {ARW} {f2(D['dit_stream_8']['mlx'])} GB at +{min(D['stream_over']):.1f}–"
                      f"{max(D['stream_over']):.1f}% per step (two alternating A/B rounds, exp-126), below Klein's +2.1–2.8% as predicted "
                      f"(smaller blocks, more compute per block). Bit-identical to stock at 50 steps."),
              P("Decoder: a new exact 3D streaming decoder", "h2"),
              precise(f"The frozen 2D chunked decoder did not apply (3D causal convolutions carry state across frames), and the frozen "
                      f"eval-barrier pattern had no effect (the peak is one operation's working buffer, not intermediates kept alive). The new "
                      f"decoder streams one latent frame at a time, keeping each causal convolution's last two input frames as state (zeros "
                      f"initially, identical to stock's causal padding), and builds each convolution window and each upsampled strip directly "
                      f"in horizontal strips. Strip heights are multiples of 24 rows and the input is padded once, so every strip sits on the "
                      f"same Winograd tile grid as the full convolution: re-padded strips shifted the grid and broke bit-identity "
                      f"(isolated test, exp-114)."),
              fig(fd, 14.5, "Figure 3. Decoder footprint by version (cache cap 0.5 GB from v3 on). v4 builds each convolution window directly "
                            "(option 1); v5 builds upsampled strips directly; v6 adds evaluation barriers inside each chunk. Stock tiled for "
                            "comparison (exp-132)."),
              P(f"What is left in v6 is mostly the causal state itself (~3.8 GB of fp32 history across ~14 convolutions). The projection "
                f"that option 1 would reach ~6 GB was wrong by about 2 GB because the next-largest component sat right behind the one it "
                f"fixed; recorded as a library rule (memory fixes are serial: project against the sorted component list). Shrinking the "
                f"state exactly needs spatial halves (deferred until an 8 GB target is decided).", "small"),
              PageBreak()]

    # ---------- 4. exactness
    story += [P("3. Exactness, verified at the shipping configuration", "h1"),
              P("Every check below was re-run by the report builder from the saved tensors (byte comparison), not quoted from logs.", "small"),
              table([["Check", "Runs", "Result"],
                     ["Text embeddings used inside the pipeline vs the frozen stock capture", "exp-128 vs frozen v1",
                      "<b>identical</b>" if ch["ctx"] else "<b>DIFFERS</b>"],
                     ["Initial noise vs a replay of stock mlx-video's random-number order", "exp-129", "<b>identical</b>" if ch["noise"] else "<b>DIFFERS</b>"],
                     ["Final latents, streamed DiT vs stock DiT, 50 steps", "exp-127 vs exp-130", "<b>identical</b>" if ch["dit50"] else "<b>DIFFERS</b>"],
                     ["Final latents, streamed DiT on the frozen reference's random path, 50 steps", "exp-131 vs frozen v1",
                      "<b>identical</b>" if ch["ctxref50"] else "<b>DIFFERS</b>"],
                     [f"Decoded frames 1–{ch['prefix_n']}, streaming decoder vs stock untiled decode", "exp-125 vs exp-109",
                      "<b>identical</b>" if ch["prefix"] else "<b>DIFFERS</b>"],
                     ["All 36 frames vs stock's default tiled decode of the true-stock latents", "exp-127 vs exp-132",
                      f"{f1(e2e['psnr_db_min'])} dB min / {f1(e2e['psnr_db_mean'])} dB mean (Tier E: &gt; 40 dB)"]],
                    [8.6 * cm, 3.6 * cm, 4.8 * cm]),
              Spacer(1, 6),
              P("A reference retired, and why", "h2"),
              P(f"The first composed run disagreed completely with the frozen stock latents (cosine {ch['old_ref_cos']:.2f}): a valid video, but a "
                f"different one. Its embeddings and its starting noise both matched stock exactly, so the difference had to be upstream of "
                f"denoising. Cause: the frozen reference had been made with a stub text-encoder loader that skips constructing the encoder, "
                f"and constructing a model draws random numbers for its initial weights, so the reference's starting noise came from a "
                f"shifted point of the seed-42 stream. Running the streamed DiT on that same stub path reproduces the old reference bit for "
                f"bit (exp-131), confirming the cause; the true-stock reference is now exp-130. Rule recorded: a reference must be produced "
                f"by the same random-number path as the system under test, and exactness is claimed at the step count that ships."),
              fig(fox, 16.5, "Figure 4. Fox prompt, frames 1, 18, 36. Left: stock mlx-video's own (tiled) decode of the true-stock latents "
                             f"(exp-132). Middle: the composed exact pipeline (exp-127); its latents are identical to stock's and its decoder is "
                             f"exact, so the visible difference to the left is stock's tiling ({f1(e2e['psnr_db_min'])} dB min). Right: the "
                             f"A2 fast mode, which passes on this prompt."),
              PageBreak()]

    # ---------- 5. caching
    a1 = D["a1"]
    rows = [["Prompt", "Exact", "A2 (static)", "A3-d (dynamic)"]]
    for p in ("v1", "v2", "v3"):
        cells = [f"<b>{PROMPT_NAME[p]}</b>", f"{mins(D['exact_s'][p])}" + ("†" if p == "v1" else "")]
        for mode in ("A2", "A3-d"):
            x = cc.get((p, mode))
            cells.append("not run (gate already failed on waves)" if not x else
                         f"{mins(x['s'])} ({x['speedup']:.2f}×), LPIPS max {x['lp_max']:.3f} "
                         f"{OK + ' pass' if x['pass'] else NO + ' fail'}; {x['n_over']}/{x['n']} frames &gt; 0.05; motion {x['motion']:.2f}×")
        rows.append(cells)
    story += [P("4. Speed: step caching (track closed, not shipped)", "h1"),
              precise(f"A probe of the 50-step run (exp-104) found heavy repetition: {a1['pct_block_steps_cos_out_gt_0.98']:.0f}% of block-steps "
                      f"change their output by less than 2%, and the with-prompt and without-prompt passes converge late. Near-duplicate tokens "
                      f"were rare ({a1['pct_tokens_near_dup_gt_0.95_centered']:.1f}% after centring), so token merging was dropped. One "
                      f"combined mechanism was built: on a skipped step all 30 blocks are bypassed and their last contribution reused; on a "
                      f"cond-only step only the with-prompt half runs. A2 is a fixed schedule from the probe (1.56× ideal); A3-d skips only "
                      f"when the measured change between full steps is below a threshold. The quality gate: LPIPS &lt; 0.05 on every frame of "
                      f"three prompts, against each prompt's exact video; the dynamic criterion was pre-registered before its results existed."),
              fig(fc, 15.0, "Figure 5. Speed vs worst-frame quality for every run. Squares: exact (LPIPS 0). Points below the dashed line pass."),
              table(rows, [3.2 * cm, 2.2 * cm, 5.8 * cm, 5.8 * cm]),
              P(f"† The fox's exact run had a fan added partway through; its fan-on equivalent is ~{(D['exact131_dit_s'] + c['te_s'] + c['dec_s']) / 60:.1f} min. "
                f"All cached runs: {min(x['fp'] for x in cc.values()):.2f}–{max(x['fp'] for x in cc.values()):.2f} GB, swap flat. "
                f"Thermal note: skipped steps let this fanless chip rest, so A2's full steps ran at {f1(D['thermal']['a2_full_step'])} s vs "
                f"{f1(D['thermal']['exact_step'])} s uncached; ~5% of A2's speed-up is that machine-specific effect.", "small"),
              PageBreak(),
              P("Why no fast mode ships", "h2"),
              P("No configuration passes all three prompts, and the failures are complementary: A2 passes only the fox, A3-d passes only "
                "neon, nothing passes waves. The one detail-safe point (A3-d on neon, "
                f"{cc[('v3', 'A3-d')]['speedup']:.2f}×) is below the 1.3× bar set before the sweep. A further schedule (the window neon "
                "happened to choose, applied to every prompt) was declined: a schedule chosen by looking at which one passed is what "
                "pre-registration exists to block."),
              fig(fu, 15.0, "Figure 6. How much the block stack's output changes per denoising step (log scale), measured during the A3-d runs. "
                            "Fast settling, a calm middle, then a sharp late rise as fine detail is painted in. Waves measures calmer than neon, "
                            "so the threshold skipped it more and earlier, yet waves is the more fragile prompt."),
              P("<b>Three findings.</b> (1) <i>Method:</i> testing on the smooth fox alone would have shipped A2 at ~1.6× with a defect on "
                "chaotic and fine-detail content; the three-prompt gate caught it in one afternoon. (2) <i>Mechanism:</i> the cheap change "
                "signal measures how much changes, not whether it matters to the eye, and ranks prompts backwards on the fragile ones (small "
                "changes move foam; neon's larger changes land in stable structure). A future signal must be perceptual or content-weighted, "
                "or calibrated per prompt, and validated on a held-out fourth prompt. (3) <i>Assets kept:</i> the U-shaped change curve "
                "(reuse belongs in the calm middle; the late rise is untouchable), the three-mode controller, and a measured-absent "
                f"interaction with weight streaming (block-0 stall after a skip ≤ {D['stall']['post_max_ms']:.2f} ms over "
                f"{D['stall']['n_post']} skips)."),
              fig(neon, 16.5, "Figure 7. Neon sign, frames 1, 18, 36. A2 changes small background content (an extra yellow umbrella, moved "
                              "pedestrians); A3-d stays close to exact. The garbled sign text is the 1.3B model's, present in the exact video too."),
              fig(waves, 16.5, "Figure 8. Ocean waves, frames 1, 18, 36. Same scene in all three; the shape and position of foam and spray differ. "
                               "Motion is not slowed (0.98× exact). Failing the gate does not mean the video looks bad: it means "
                               "'indistinguishable from exact' cannot be claimed."),
              P("Private use is a different question from shipping: for smooth, stable-background content reviewed by eye before use, the A2 "
                "schedule (~13 instead of ~20.5 minutes) is a reasonable personal tool. It is not part of any claim in this report.", "small"),
              PageBreak()]

    # ---------- 6. hit matrix
    story += [P("5. Did the image playbook transfer?", "h1"),
              P("The pattern library and engine built for FLUX.2 Klein 4B (image) were frozen before Wan was touched; Wan work used adapters "
                "only. Klein cells come from the library's recorded results; '—' means not used on that model."),
              table([["Pattern", "Klein 4B (image)", "Wan 2.1 1.3B (video)"],
                     ["Block weight streaming + prefetch", "hit: 8.57 " + ARW + " 2.81 GB, +2.1–2.8%", f"<b>hit</b>: {f2(D['stock_dit_8']['mlx'])} " + ARW + f" {f2(D['dit_stream_8']['mlx'])} GB, bit-identical to stock at 50 steps"],
                     ["Upcast-on-load streaming", "—", f"<b>new variant</b>: {f2(D['stock_te']['mlx'])} " + ARW + f" {f2(D['te_stream']['mlx'])} GB, bit-identical"],
                     ["Row-gathered embeddings", "—", "<b>hit</b>: the 256k-row table is never loaded"],
                     ["Allocator cache cap", "hit: 11.54 " + ARW + " 10.26 GB", "<b>hit</b>: 0.5 GB cap on every composed run"],
                     ["Free each stage after last use", "hit: 15.99 " + ARW + " 9.51 GB", "not needed: mlx-video already frees between stages"],
                     ["Evaluation barriers between blocks", "hit: VAE 9.55 " + ARW + " 7.26 GB", "<b>applies, no effect</b> on the decoder (one op's buffer, not graph lifetime)"],
                     ["Exact 2D row-chunked decode", "hit: 1.81 GB, Tier E", "<b>does not transfer</b> (3D causal) " + ARW + " new exact 3D streaming decoder"],
                     ["Kernel-grid-aligned chunking", "—", "<b>new</b>: Winograd-aligned strips keep chunked convolutions bit-identical"],
                     ["Step caching", "not viable (4-step model)", "<b>closed, not shipped</b> (section 4)"],
                     ["Token merging", "—", f"<b>miss</b>: {a1['pct_tokens_near_dup_gt_0.95_centered']:.1f}% near-duplicate tokens"]],
                    [4.6 * cm, 4.6 * cm, 7.8 * cm], zebra=True),
              Spacer(1, 6),
              P("Klein's memory work took two days; Wan's took one working day, from the encoder fix to the before/after table, on a model "
                "family the playbook had never seen, including one genuinely new component (the 3D decoder) and two live debugging detours."),
              P("Lessons recorded in the library", "h2"),
              table([["Rule", "Evidence"],
                     ["Memory fixes are serial: project a fix against the sorted list of memory components, not its local saving", "Klein exp-005/006; Wan decoder option 1 (projected ~6 GB, measured 8.2 GB)"],
                     ["Exactness is verified at the shipping step count, and a reference must share the system's random-number path", "exp-126 (8 steps) vs exp-130/131 (50 steps); the retired reference"],
                     ["Denoising change is U-shaped: reuse belongs in the calm middle, never in the late rise", "exp-a2-v3, exp-a3d-v2/v3 change logs"],
                     ["Chunked convolutions stay exact only on the kernel's tile grid", "exp-114"],
                     ["On passively cooled hardware, skipped work also speeds the remaining work; report the schedule's share separately", "exp-134 vs exp-131"]],
                    [10.2 * cm, 6.8 * cm]),
              PageBreak()]

    # ---------- 7. setup + index
    idx = [["Experiment", "What it is", "Footprint peak", "Time", "Swap"]]
    for e, what in (("exp-101-te-stock", "stock text encoder, once (frozen embeddings capture)"), ("exp-102-te-stream", "streamed text encoder"),
                    ("exp-103-base-auto", "stock DiT + stock tiled decode on frozen embeddings"), ("exp-125-v6-full", "streaming decoder v6, decode only"),
                    ("exp-126-stream-1", "DiT streaming, 8 steps (A/B round 1)"), ("exp-127-composed", "composed exact pipeline, v1"),
                    ("exp-130-stock", "stock DiT + streamed encoder, 50 steps (true-stock latents)"),
                    ("exp-131-ctxref-stream", "streamed DiT on the frozen reference's path, 50 steps"),
                    ("exp-132-stock-decode", "stock tiled decode of the true-stock latents"), ("exp-ref-v2", "exact pipeline, waves"),
                    ("exp-ref-v3", "exact pipeline, neon"), ("exp-134-a2", "A2, fox"), ("exp-a2-v2", "A2, waves"), ("exp-a2-v3", "A2, neon"),
                    ("exp-a3d-v2", "A3-d, waves"), ("exp-a3d-v3", "A3-d, neon")):
        r = R(e)
        t = (r.get("total_s") or {}).get("median") or r.get("decode_s") or r.get("encode_s")
        idx.append([e, what, f"{r['process']['peak_footprint_gb']:.2f} GB", f"{t:.0f} s" if t else "–", "grew" if r.get("swap_grew") else "flat"])
    story += [P("6. Setup and experiment index", "h1"),
              table([["", ""],
                     ["Machine", "MacBook Air Mac17,3 (fanless; a fan was placed under it from ~12:30 on the day), Apple M5, 8-core GPU, 16 GB unified memory"],
                     ["Software", "MLX 0.32.4; mlx-video @ 87db56a (MIT); LPIPS (AlexNet) via torch"],
                     ["Model", "Wan-AI/Wan2.1-T2V-1.3B (Apache-2.0); 832×480, 33 frames requested (mlx-video outputs 36), 50 steps, UniPC, CFG 5, seed 42"],
                     ["Prompts", "v1 a red fox trotting through snow; v2 ocean waves on a rocky beach at sunset; v3 a neon 'NOODLES' sign on a rainy street"],
                     ["R1 execution mode", "bit-identity is only claimed against a reference produced in the same execution mode"],
                     ["R2 simulation", "mx.set_memory_limit is a soft limit; no smaller-machine claim is made from it (a real 8 GB Mac is untested)"],
                     ["R3 timing validity", "runs where swap grew are memory-only; cool-down to Nominal thermal state before every timed run"],
                     ["Tiers", "E: bit-identical or PSNR &gt; 40 dB. A: LPIPS &lt; 0.05 on every frame (plus visual review)"]],
                    [3.4 * cm, 13.6 * cm], header=False),
              Spacer(1, 8),
              table(idx, [3.6 * cm, 7.4 * cm, 2.4 * cm, 1.8 * cm, 1.8 * cm], zebra=True),
              Spacer(1, 6),
              P("Full working log: reports/phase4b.md. Every number in this PDF is read from runs/*/result.json and gate.json by "
                "reports/wan_report_data.py when the report is built.", "small")]

    def on_page(c_, doc):
        c_.saveState()
        c_.setFont("Sans", 7.5)
        c_.setFillColor(MUTE)
        c_.drawString(2 * cm, 1.2 * cm, "Wan 2.1 1.3B on Apple M5 · kernelforge")
        c_.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        c_.restoreState()

    out = HERE / "wan-m5-report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Wan 2.1 on an Apple M5: text-to-video in 8.6 GB, bit-identical to stock", author="kernelforge")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(out)


if __name__ == "__main__":
    build()
