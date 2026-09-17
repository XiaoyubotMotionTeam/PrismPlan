# -*- coding: utf-8 -*-
"""plot_scaling — turn ``bench_scaling.py`` sweeps into publication figures.

Consumes the ``scaling_summary.json`` files written by
:mod:`examples.bench_scaling` for the three swept knobs and emits the honest
intra-solve scaling figures used in the paper:

* ``scaling_width.pdf`` — median solve time vs. ``num_new_configs`` (parallel
  tree-extension width) on a log-x axis, over the flat planner memory footprint
  on a second, stacked panel sharing that axis. This is *intra-solve parallel
  width*, NOT concurrent batch size B (the shipped package has no batch-of-B
  API).
* ``mem_ceiling.pdf`` — planner device memory over median solve time vs.
  ``max_samples`` (per-tree node capacity), log-log. Shows the memory ceiling.
* ``ablation_granularity.pdf`` — success rate over median solve time vs.
  ``granularity`` (waypoints checked per edge), the only runtime-toggleable
  backend knob.

Each panel carries exactly one y scale: two measures of different units are
stacked on a shared x axis rather than folded onto a twin axis, so no reader has
to work out which curve belongs to which side.  Typography, palette and geometry
come from :mod:`examples.paperstyle`, shared with :mod:`examples.plot_mbm`.

numpy + matplotlib only; no ``prrtc`` / CUDA import, so it runs anywhere the
JSON does. Usage::

    python examples/plot_scaling.py --width results/scaling/width \\
        --capacity results/scaling/capacity \\
        --granularity results/scaling/granularity --outdir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import paperstyle as ps  # noqa: E402

# One measure, one hue: time reads in the sampling blue (these sweeps are all
# pRRTC), memory and success rate in the two remaining paradigm hues, so a
# color never means two things across the figure set.
_TIME = ps.C_SAMPLING_D
_MEM = ps.C_OPT_D
_RATE = ps.C_SEARCH_D


def _load(indir):
    return json.load(open(Path(indir) / "scaling_summary.json"))


def _series(summary):
    """Return (values, median_ms, planner_mem_mb, success_rate) arrays."""
    rows = summary["by_value"]
    v = np.array([r["value"] for r in rows], dtype=float)
    med = np.array([r["solve_ms"]["median"] for r in rows], dtype=float)
    mem = np.array([r["planner_mem_mb"] for r in rows], dtype=float)
    sr = np.array([r["success_rate"] for r in rows], dtype=float)
    return v, med, mem, sr


def _stacked(width, height=2.55):
    """Two panels, shared x, 2:1 heights — the primary measure on top."""
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(width, height), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]}, layout="constrained")
    return fig, axT, axB


def plot_width(summary, outpath):
    v, med, mem, _ = _series(summary)
    fig, axT, axB = _stacked(ps.W_HALF)

    axT.plot(v, med, "o-", color=_TIME, markeredgecolor="white")
    axT.set_ylabel("median solve time (ms)")
    axT.set_title("(a) parallel width vs. solve time", loc="left")
    ps.grid_both(axT)
    imin = int(np.argmin(med))
    axT.annotate(f"min {med[imin]:.1f} ms\n@ {int(v[imin])}",
                 xy=(v[imin], med[imin]), xytext=(0.40, 0.72),
                 textcoords="axes fraction", fontsize=6.5, color=ps.INK_2,
                 arrowprops=dict(arrowstyle="->", lw=0.5, color=ps.INK_2,
                                 shrinkB=3))

    axB.plot(v, mem, "s-", color=_MEM, markeredgecolor="white")
    axB.set_ylabel("memory (MiB)")
    axB.set_ylim(0, max(mem) * 1.6)
    axB.set_xscale("log", base=2)
    axB.set_xlabel("parallel width $W$")
    ps.grid_both(axB)

    ps.save(fig, outpath)


def plot_mem_ceiling(summary, outpath):
    v, med, mem, _ = _series(summary)
    fig, axT, axB = _stacked(ps.W_HALF)

    axT.plot(v, mem, "s-", color=_MEM, markeredgecolor="white")
    axT.set_yscale("log")
    axT.set_ylabel("planner memory (MiB)")
    axT.set_title("(b) tree capacity vs. memory", loc="left")
    ps.grid_both(axT)

    axB.plot(v, med, "o-", color=_TIME, markeredgecolor="white")
    axB.set_yscale("log")
    axB.set_ylabel("solve (ms)")
    axB.set_xscale("log")
    axB.set_xlabel("tree capacity (nodes)")
    ps.grid_both(axB)

    ps.save(fig, outpath)


def plot_granularity(summary, outpath):
    v, med, mem, sr = _series(summary)
    fig, axT, axB = _stacked(ps.W_5, height=2.6)
    x = np.arange(len(v))

    axT.plot(x, 100.0 * sr, "D-", color=_RATE, markeredgecolor="white", ms=3.2)
    axT.set_ylabel("success rate (%)")
    axT.set_ylim(min(60.0, 100.0 * min(sr) - 8), 108)
    ps.grid_both(axT)
    for xi, s in zip(x, sr):
        ps.label(axT, xi, 100.0 * s, f"{100 * s:.0f}%", dy=1.2)

    axB.bar(x, med, width=0.6, color=_TIME, edgecolor=ps.INK, linewidth=0.5)
    axB.set_ylabel("solve (ms)")
    axB.set_xticks(x)
    axB.set_xticklabels([str(int(vi)) for vi in v])
    axB.set_xlabel("granularity (waypoints per edge)")
    ps.grid_y(axB)

    ps.save(fig, outpath)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", required=True)
    ap.add_argument("--capacity", required=True)
    ap.add_argument("--granularity", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    ps.apply()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_width(_load(args.width), outdir / "scaling_width.pdf")
    plot_mem_ceiling(_load(args.capacity), outdir / "mem_ceiling.pdf")
    plot_granularity(_load(args.granularity), outdir / "ablation_granularity.pdf")
    print(f"wrote scaling_width.pdf, mem_ceiling.pdf, ablation_granularity.pdf "
          f"to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
