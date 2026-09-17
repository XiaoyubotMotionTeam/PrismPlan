# Benchmark provenance and reproduction

The README figures reflect the manuscript and artifacts at source revision
`c9e9014c7eb6198f28e0b81b23c5ab870caa529d` (2026-09-16). They were regenerated
from the recorded run, not from a new benchmark on the release maintainer's
machine. This source release includes the benchmark and plotting tools and the
README images; it excludes the manuscript, full datasets, and raw run outputs.

## Reference runs

| Run identifier | Protocol | Role |
|---|---|---|
| `mbm_800_flag` | Panda, 800 Robometrics MotionBenchMaker problems, 1 trial per planner/problem, 2000 ms per planner, RTX 3060 | Paper's main Panda comparison and both README result figures |
| `mbm_800_1s` | Same dataset and implementation, 1000 ms per planner | Budget-sensitivity control |

The older `mbm_800` run predates the removal of restrictive iteration caps and
is not the paper's one-second control. Its results must not be substituted for
either reference run.

| Planner | Solved / 800 | Overall success | Median solve time on solved problems (ms) |
|---|---:|---:|---:|
| pRRTC | 785 | 98.1% | 11.0 |
| MIT* | 616 | 77.0% | 2008.8 |
| MHA* | 722 | 90.2% | 3.4 |
| wPA*SE | 756 | 94.5% | 3.0 |
| STOMP | 617 | 77.1% | 18.9 |
| CHOMP | 80 | 10.0% | 3.7 |

Fifteen goals are already in collision. Overall success uses all 800 problems;
feasible success uses 785. Timing medians use each planner's own successful
requests, not a common solved subset. A configured time limit is a termination
budget, not a hard real-time deadline: measured latency can exceed it slightly.
The synchronized trajectory videos use separate individual runs and are
qualitative demonstrations, not this equal-budget aggregate experiment.

## Run and plot

Build the CUDA module and set `PYTHONPATH` as described in the root README.
The full Panda dataset requires [fishbotics/robometrics](https://github.com/fishbotics/robometrics)
with its `geometrout` dependency, not the unrelated PyPI package with the same
name. Install it in the active environment. From `code/gpu_planning`:

```bash
python3 -m pip install "git+https://github.com/fishbotics/robometrics.git@81e3d1d605de84100d8ab880b43096aba221a48b" geometrout

python3 examples/bench_mbm.py \
  --source robometrics --dataset mbm \
  --budget-ms 2000 --trials 1 --out results/mbm_800_flag

python3 examples/plot_mbm.py \
  --indir results/mbm_800_flag --outdir results/mbm_800_flag/figures
```

The runner writes `trials.csv` and `summary.json`; the latter records
`config.budget_ms`. Omitting `--budget-ms` preserves per-planner YAML defaults
(4000 ms for pRRTC, 1000 ms for the others) and records `null`, which is not an
equal-budget run. Use 1000 ms and a separate output directory for the control.
Check that the full Panda run loads 800 problems. The bundled three-problem
sample checks the workflow only.

The plotting scripts depend on NumPy and Matplotlib and can run without CUDA.
Both import the adjacent `paperstyle.py`, which keeps their palette and
typography consistent. To export a new run's figures as README PNGs, use Poppler
from the same directory:

```bash
for name in mbm_cross_paradigm mbm_scene_heatmap; do
  pdftocairo -png -r 240 -singlefile \
    "results/mbm_800_flag/figures/$name.pdf" \
    "../../docs/assets/results/$name"
done
```

When replacing the reference images with a new run, update this provenance and
the README numbers together. Hardware, host load, and stochastic execution can
change the exact success counts and timings; a rerun should not be presented as
the original recorded experiment.
