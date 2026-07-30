# `hpc/` — BlueBEAR job scripts

Environment setup lives in [`env.sh`](env.sh) (modules + venv); every job sources it.
See also [`setup_data.sh`](setup_data.sh) for unpacking raw DSEC.

## Adversarial attack pipeline

Computes the full attack dataset (clean / FGSM / PGD / random-sign control, over an
epsilon range × objective, with gradient attribution) across the DSEC test sequences.

**Steps**

1. **Calibrate epsilon** (CPU, ~minutes) — [`calibrate_epsilon.slurm`](calibrate_epsilon.slurm).
   Scans the data and writes the epsilon list to `results/calibrated_epsilons.txt`.
2. **Sweep** (GPU) — [`sweep_attack.slurm`](sweep_attack.slurm), one job per attack.
   Reads that file (prepending `0.0` as the clean baseline) and runs `sweep_epsilon.py`
   over the split folder with `--attribution`.

**One command** (chains calibrate → fgsm + pgd via a Slurm dependency):

```bash
bash hpc/submit_attacks.sh          # full run
SMOKE=1 bash hpc/submit_attacks.sh  # 5 samples/sequence first, to check end-to-end
```

## The split

Default `SPLIT=valid_split_doubleseq.csv` — the paper's 5 validation/test sequences
(`thun_00_a, zurich_city_02_d, zurich_city_03_a, zurich_city_08_a, zurich_city_11_b`,
2152 samples), already preprocessed for the baseline. Output is split **per sequence**
automatically (the sequence is derived from each sample's filename), so no per-sequence
input CSVs are needed. `SPLIT` may also be a folder, in which case every `*.csv` in it
is swept. Override with `SPLIT=... bash hpc/submit_attacks.sh`.

## Outputs (per attack `A`, under `results/sweep_A/`)

| file | contents |
|------|----------|
| `raw_A_<sequence>.csv` | every sample × condition row (incl. each random draw) |
| `per_sequence_A.csv` | pooled mean+std per sequence/condition/objective/epsilon |
| `sweep_A.csv` | split-level aggregate curve |
| `attribution_A.csv` | per-sample temporal (T) + polarity (ON/OFF) gradient profiles |
| `gradmap_A_<objective>_<sequence>.npy` | per-sequence mean spatial heatmap (H×W) |

Job logs go to `hpc/logs/`. Tune per-attack with env vars (see the header of
`sweep_attack.slurm`): `LOSSES`, `RAND_RESTARTS`, `EXTRA_ARGS` (e.g. `--iters 7` for PGD).
