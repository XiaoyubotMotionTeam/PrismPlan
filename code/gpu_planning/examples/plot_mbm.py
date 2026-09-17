# -*- coding: utf-8 -*-
"""plot_mbm — turn a ``bench_mbm.py`` run into publication figures.

Consumes the ``trials.csv`` + ``summary.json`` written by
:mod:`examples.bench_mbm` and emits two vector (PDF) figures used in the paper's
cross-paradigm section:

* ``mbm_cross_paradigm.pdf`` — two panels: per-planner success rate over the
  whole problem set, and median solve time on the trials that *succeeded*
  (log scale). Fill hue and hatch angle encode the paradigm.
* ``mbm_scene_heatmap.pdf`` — a planner x scene grid of success rate on a
  single-hue magnitude ramp, exposing which scene families each paradigm
  handles.

Typography, palette and geometry come from :mod:`examples.paperstyle`, which is
shared with :mod:`examples.plot_scaling` so both figures match the paper body.

numpy + matplotlib only; no ``prrtc`` / CUDA import, so it runs anywhere the CSV
does. Usage::

    python examples/plot_mbm.py --indir /tmp/mbm_800 --outdir figures
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import paperstyle as ps  # noqa: E402


def _read_trials(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["success"] = str(r["success"]).lower() in ("true", "1", "yes")
        for k in ("solve_ms", "cost"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = float("nan")
    return rows


def plot_success_and_time(summary, trials, outpath):
    present = set(summary["by_planner"])
    planners = ps.order(present)

    rate = [100.0 * summary["by_planner"][p]["all"]["success_rate"]
            for p in planners]
    succ = [summary["by_planner"][p]["all"]["successes"] for p in planners]
    att = [summary["by_planner"][p]["all"]["attempts"] for p in planners]

    # Median solve time over SUCCESSFUL trials only (fair latency signal;
    # failures that burn the full budget would otherwise dominate the mean).
    med_ms = []
    by_pl = defaultdict(list)
    for r in trials:
        if r["success"]:
            by_pl[r["planner"]].append(r["solve_ms"])
    for p in planners:
        v = by_pl.get(p, [])
        med_ms.append(float(np.median(v)) if v else float("nan"))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(ps.W_FULL, 2.35),
                                   layout="constrained")
    x = np.arange(len(planners))

    for xi, p in zip(x, planners):
        axL.bar(xi, rate[xi], width=0.68, **ps.bar_kw(p))
    for xi, (r, s, a) in enumerate(zip(rate, succ, att)):
        ps.label(axL, xi, r, f"{s}/{a}", dy=1.5)
    axL.set_ylabel("success rate (%)")
    axL.set_ylim(0, 112)
    axL.set_yticks([0, 20, 40, 60, 80, 100])
    axL.set_xticks(x)
    axL.set_xticklabels(planners, rotation=25, ha="right")
    axL.set_title("(a) success rate", loc="left")
    ps.grid_y(axL)
    ps.paradigm_legend(axL, loc="upper right", bbox_to_anchor=(1.0, 1.02))

    for xi, p in zip(x, planners):
        axR.bar(xi, med_ms[xi], width=0.68, **ps.bar_kw(p))
    axR.set_yscale("log")
    axR.set_ylabel("median solve time (ms), solved only")
    axR.set_xticks(x)
    axR.set_xticklabels(planners, rotation=25, ha="right")
    axR.set_title("(b) latency on solved problems", loc="left")
    ps.grid_y(axR)
    for xi, m in enumerate(med_ms):
        if np.isfinite(m):
            ps.label(axR, xi, m * 1.15, f"{m:.0f}")

    ps.save(fig, outpath)
    return planners, rate, med_ms


def plot_scene_heatmap(trials, outpath):
    scenes = sorted({r["scene"] for r in trials})
    present = {r["planner"] for r in trials}
    planners = ps.order(present)

    succ = defaultdict(lambda: defaultdict(int))
    att = defaultdict(lambda: defaultdict(int))
    for r in trials:
        att[r["planner"]][r["scene"]] += 1
        if r["success"]:
            succ[r["planner"]][r["scene"]] += 1

    M = np.full((len(planners), len(scenes)), np.nan)
    for i, p in enumerate(planners):
        for j, s in enumerate(scenes):
            a = att[p][s]
            if a:
                M[i, j] = succ[p][s] / a

    fig, ax = plt.subplots(figsize=(ps.W_FULL, 2.15), layout="constrained")
    im = ax.imshow(M, cmap=ps.SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels([s.replace("_panda", "")
                        for s in scenes], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(planners)))
    ax.set_yticklabels(planners)
    ax.set_xticks(np.arange(len(scenes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(planners) + 1) - 0.5, minor=True)
    # a hairline surface gap between cells, so adjacent fills never touch
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(planners)):
        for j in range(len(scenes)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{succ[planners[i]][scenes[j]]}/"
                              f"{att[planners[i]][scenes[j]]}",
                        ha="center", va="center", fontsize=6.0,
                        color="white" if M[i, j] > ps.SEQ_FLIP else ps.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("success rate", fontsize=7.0)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=1.6, width=0.4)
    ax.set_title("per-scene success rate (shared backend, identical problems)",
                 loc="left")
    ps.save(fig, outpath)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True,
                    help="dir with trials.csv + summary.json from bench_mbm.py")
    ap.add_argument("--outdir", required=True, help="dir to write PDF figures")
    args = ap.parse_args(argv)

    ps.apply()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(indir / "summary.json"))
    trials = _read_trials(indir / "trials.csv")

    f1 = outdir / "mbm_cross_paradigm.pdf"
    f2 = outdir / "mbm_scene_heatmap.pdf"
    planners, rate, med = plot_success_and_time(summary, trials, f1)
    plot_scene_heatmap(trials, f2)

    cfg = summary.get("config", {})
    print(f"n_problems={cfg.get('n_problems')} trials={cfg.get('trials')} "
          f"robot={cfg.get('robot')}")
    for p, r, m in zip(planners, rate, med):
        print(f"  {p:8s} success={r:5.1f}%  median_solved_ms={m:8.1f}")
    print(f"wrote {f1}\nwrote {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
