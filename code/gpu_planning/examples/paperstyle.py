# -*- coding: utf-8 -*-
"""paperstyle — one source of truth for the look of every figure in the paper.

Both plotting scripts (:mod:`examples.plot_mbm`, :mod:`examples.plot_scaling`)
import this module and call :func:`apply` before creating a figure, so the paper
ships a single typographic and chromatic system instead of per-script defaults.

Typography.  The paper body is 10pt Times (``arxiv.sty`` sets ``\\rmdefault``
to ``ptm``), so figures use a metric-compatible Times clone at 8pt with STIX
math.  ``pdf.fonttype = 42`` embeds TrueType
outlines rather than matplotlib's default Type 3, which several venues reject
and arXiv flags.

Geometry.  Figures are authored at their *final* printed width (see ``W_*``
below, from ``textwidth = 6.5in``) and saved without ``bbox_inches="tight"``,
so LaTeX includes them at scale 1.0 and 8pt in the figure really is 8pt on the
page.  Use ``layout="constrained"`` to fit labels inside that fixed canvas.

Color.  Hue encodes the *paradigm*, lightness the planner within it, and a hatch
angle repeats the paradigm as a non-color channel.  The six fills were validated
against the white page for the OKLCH lightness band, the chroma floor, adjacent
protan/deutan separation (worst pair ΔE 14.7, target >= 8), the normal-vision
floor (worst pair 17.4, floor 15) and contrast; the previous ad-hoc palette
failed three of those checks (notably STOMP vs. wPA*SE at protan ΔE 3.5).
The magnitude ramp is a single blue hue, light->dark, replacing the earlier
red-yellow-green map that encoded magnitude with the one hue pair colorblind
readers cannot separate.  Keep these hexes in sync with the ``pp*`` colors
defined in the paper preamble.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ── printed widths (inches), from textwidth = 6.5in ───────────────────────────
W_FULL = 6.5      # \includegraphics[width=\linewidth]
W_HALF = 3.185    # width=0.49\linewidth  (two side by side)
W_5    = 3.575    # width=0.55\linewidth
W_7    = 4.68     # width=0.72\linewidth

# ── ink (text and chrome never wear a series color) ──────────────────────────
INK      = "#111111"   # primary text, mark outlines
INK_2    = "#555555"   # secondary text: value labels, axis labels
INK_MUTE = "#777777"   # tick labels
GRID     = "#D9D9D9"   # hairline grid
AXIS     = "#BBBBBB"   # spines / baseline

# ── categorical fills: hue = paradigm, lightness = planner within paradigm ───
C_SAMPLING_D, C_SAMPLING_L = "#1c5cab", "#5f9fe8"
C_SEARCH_D,   C_SEARCH_L   = "#ad4f18", "#e08a4e"
C_OPT_D,      C_OPT_L      = "#1a7346", "#4faf7d"

H_SAMPLING, H_SEARCH, H_OPT = "", "//", "\\\\"

# planner -> (paradigm label, fill, hatch).  Fixed order, never cycled.
PLANNER = {
    "pRRTC":  ("sampling",            C_SAMPLING_D, H_SAMPLING),
    "MIT*":   ("anytime-optimal",     C_SAMPLING_L, H_SAMPLING),
    "MHA*":   ("search",              C_SEARCH_D,   H_SEARCH),
    "wPA*SE": ("parallel search",     C_SEARCH_L,   H_SEARCH),
    "STOMP":  ("optimization (DF)",   C_OPT_D,      H_OPT),
    "CHOMP":  ("optimization (grad)", C_OPT_L,      H_OPT),
}
ORDER = ["pRRTC", "MIT*", "MHA*", "wPA*SE", "STOMP", "CHOMP"]
_FALLBACK = ("other", "#8A8A8A", "")

# single-hue magnitude ramp (light = 0, dark = 1)
SEQ_STEPS = ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec",
             "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("pp_blue", SEQ_STEPS)
SEQ_FLIP = 0.62   # cell value above which the in-cell label flips to white


def color(planner: str) -> str:
    return PLANNER.get(planner, _FALLBACK)[1]


def hatch(planner: str) -> str:
    return PLANNER.get(planner, _FALLBACK)[2]


def order(present) -> list:
    """Planners in paradigm order, keeping only those present in the data."""
    return [p for p in ORDER if p in present] + \
           [p for p in present if p not in ORDER]


def paradigm_legend(ax, **kw):
    """Legend that states what hue means, so color is never the only cue."""
    handles = [
        Patch(facecolor=C_SAMPLING_D, hatch=H_SAMPLING, edgecolor=INK,
              linewidth=0.5, label="sampling"),
        Patch(facecolor=C_SEARCH_D, hatch=H_SEARCH, edgecolor=INK,
              linewidth=0.5, label="search"),
        Patch(facecolor=C_OPT_D, hatch=H_OPT, edgecolor=INK,
              linewidth=0.5, label="optimization"),
    ]
    kw.setdefault("loc", "lower left")
    kw.setdefault("ncol", 3)
    kw.setdefault("handlelength", 1.5)
    kw.setdefault("handleheight", 0.9)
    kw.setdefault("columnspacing", 1.0)
    kw.setdefault("borderpad", 0.2)
    return ax.legend(handles=handles, **kw)


def bar_kw(planner: str) -> dict:
    """Fill + hatch + hairline outline for one planner's bars."""
    return dict(color=color(planner), hatch=hatch(planner),
                edgecolor=INK, linewidth=0.5)


def label(ax, x, y, text, *, va="bottom", dy=0.0, **kw):
    """A direct value label in secondary ink (never in the series color)."""
    kw.setdefault("ha", "center")
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("color", INK_2)
    return ax.text(x, y + dy, text, va=va, **kw)


def apply() -> None:
    """Install the paper's figure style into matplotlib's rcParams."""
    plt.rcParams.update({
        # fonts: match the 10pt Times body, embed TrueType (not Type 3)
        "font.family": "serif",
        # Liberation Serif first on purpose: it is a TrueType, metric-compatible
        # Times clone, so ``pdf.fonttype = 42`` embeds a real TrueType program.
        # The .otf Times clones (TeX Gyre Termes, Nimbus Roman) are CFF, which
        # matplotlib wraps in a TrueType /Subtype and PDF validators flag as a
        # "mismatch between font type and embedded font file".
        "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman",
                       "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 8.5,
        # ink
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_MUTE,
        "ytick.labelcolor": INK_MUTE,
        # recessive chrome, thin marks
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.6,
        "hatch.linewidth": 0.4,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "legend.frameon": False,
        "legend.borderaxespad": 0.2,
        # output: exact canvas so LaTeX includes at scale 1.0
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
    })


def grid_y(ax) -> None:
    ax.grid(axis="y", which="major")
    ax.set_axisbelow(True)


def grid_both(ax) -> None:
    ax.grid(axis="both", which="major")
    ax.set_axisbelow(True)


def save(fig, path) -> None:
    """Save at exactly ``figsize`` inches (no tight-bbox rescaling)."""
    fig.savefig(path)
    plt.close(fig)
