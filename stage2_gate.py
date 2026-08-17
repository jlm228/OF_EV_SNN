"""Compare CARLA EPE against reproduced DSEC EPE, per model, and report a pass/fail.

    python stage2_gate.py --capture 50761933

Reads runs/reproduction/reproduction.csv (the reference) and the *_metrics.csv files
score_flow.py wrote for each model. The reference is the reproduced number rather than the
published one: a reproduction shortfall and a CARLA domain gap are indistinguishable from the
CARLA side.

Reported under three masks:

  dsec_matched  the criterion. A DSEC-shaped region, so CARLA is scored over a comparable pixel
                population -- DSEC's validity mask omits the sky and most of the near-field
                road, where flow magnitude peaks.
  dense         every valid pixel. Context only; scored over higher-magnitude pixels than DSEC.
  ped           the pedestrian silhouette. No DSEC equivalent, so also context only.

Criteria, both on dsec_matched: CARLA EPE <= PASS_RATIO x reproduced DSEC EPE, and CARLA EPE
below mean |GT| so the prediction beats predicting zero flow everywhere.

A per-|GT|-magnitude breakdown is printed when the epe_vs_magnitude CSVs are present, since
matching the masks removes most but not all of the population difference.

Exit codes: 0 all pass, 1 at least one fails, 2 not every model has been run.
"""
import argparse
import glob
import os

import pandas as pd

PASS_RATIO = 2.0
# The pedestrian region is flagged, not gated, above this multiple of the gated figure.
PED_CAUTION_RATIO = 2.0

# CARLA metrics-file prefix -> the reproduction row it is compared against. The `full` variant
# throughout, because the CARLA evaluation runs at 480x640 for every model.
MODELS = [
    ("of_ev_snn", "OF_EV_SNN", "full"),
    ("snn", "SDformerFlow-SPE-QK-s10-c2", "full"),
    ("ann", "STTFlowNet-en4-b2-p4-w10", "full"),
]


def read_metrics(path):
    """metric -> float. The CSV's first row is the dataset name, so the value column parses as
    object; coerce and drop non-numeric entries."""
    d = pd.read_csv(path).set_index("metric")["value"]
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float)) or str(v).replace(".", "", 1).lstrip("-").isdigit()}


def magnitude_table(carla_csv, dsec_csv, mask="dsec_matched", dsec_mask="dsec", min_px=1000):
    """EPE per |GT| bin, CARLA against DSEC, for bins both populate."""
    if not (os.path.exists(carla_csv) and os.path.exists(dsec_csv)):
        return None
    c = pd.read_csv(carla_csv)
    d = pd.read_csv(dsec_csv)
    c = c[(c["mask"] == mask) & (c["n_pixels"] >= min_px)]
    d = d[(d["mask"] == dsec_mask) & (d["n_pixels"] >= min_px)]
    if c.empty or d.empty:
        return None
    m = c.merge(d, on=["bin_lo", "bin_hi"], suffixes=("_carla", "_dsec"))
    return m if not m.empty else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="50761933")
    ap.add_argument("--carla-eval", default=os.path.join(here, "results", "carla_eval"))
    ap.add_argument("--reproduction", default=os.path.join(here, "runs", "reproduction",
                                                           "reproduction.csv"))
    ap.add_argument("--dsec-metrics", default=None,
                    help="a DSEC sequence scored through the same pipeline, for the "
                         "magnitude breakdown (default: newest dsec_*_metrics.csv in --carla-eval)")
    args = ap.parse_args()

    repro = pd.read_csv(args.reproduction)

    dsec_metrics_path = args.dsec_metrics
    if dsec_metrics_path is None:
        found = sorted(glob.glob(os.path.join(args.carla_eval, "dsec_*_metrics.csv")))
        dsec_metrics_path = found[-1] if found else None

    print("capture      : %s" % args.capture)
    print("reference    : %s" % os.path.relpath(args.reproduction, here))
    unaudited = repro[repro["source"] == "notes"]["model"].tolist()
    if unaudited:
        print("               WARNING: %d row(s) still source=notes (no run id): %s"
              % (len(unaudited), ", ".join(sorted(set(unaudited)))))
    print()

    verdicts, missing = {}, []

    for prefix, model, variant in MODELS:
        row = repro[(repro["model"] == model) & (repro["variant"] == variant)]
        if row.empty:
            print("%-28s no reproduction row for variant %r -- skipped" % (model, variant))
            continue
        dsec_epe = float(row.iloc[0]["epe"])

        metrics_path = os.path.join(args.carla_eval, "%s_%s_metrics.csv" % (prefix, args.capture))
        print("=" * 78)
        print("%s  (reproduced DSEC EPE %.3f px, %s)" % (model, dsec_epe, variant))
        print("=" * 78)

        if not os.path.exists(metrics_path):
            print("  no CARLA metrics at %s" % os.path.relpath(metrics_path, here))
            print("  -> NOT RUN\n")
            missing.append(model)
            continue

        m = read_metrics(metrics_path)
        print("  %-14s %9s %9s %9s %9s" % ("mask", "EPE", "mean|GT|", "vs DSEC", "vs zero"))
        for mask in ("dsec_matched", "dense", "ped"):
            if "EPE_%s" % mask not in m:
                continue
            epe, gtmag = m["EPE_%s" % mask], m["gtmag_%s" % mask]
            print("  %-14s %9.3f %9.3f %8.2fx %8.2fx%s"
                  % (mask, epe, gtmag, epe / dsec_epe, epe / gtmag,
                     "   <- gate" if mask == "dsec_matched" else ""))

        if "EPE_dsec_matched" not in m:
            print("  no dsec_matched mask -- cannot gate\n")
            missing.append(model)
            continue

        epe = m["EPE_dsec_matched"]
        ratio_ok = epe <= PASS_RATIO * dsec_epe
        beats_zero = epe < m["gtmag_dsec_matched"]
        verdicts[model] = ratio_ok and beats_zero

        print()
        print("  within %.0fx reproduced DSEC EPE (%.3f <= %.3f) : %s"
              % (PASS_RATIO, epe, PASS_RATIO * dsec_epe, "yes" if ratio_ok else "NO"))
        print("  beats the zero-flow control (%.3f < %.3f)      : %s"
              % (epe, m["gtmag_dsec_matched"], "yes" if beats_zero else "NO"))
        print("  -> %s" % ("PASS" if verdicts[model] else "FAIL"))

        # The criterion is an aggregate over a mask dominated by low-magnitude pixels, so it
        # can pass while the pedestrian region is much worse. Not a failure -- there is no DSEC
        # pedestrian region to compare against -- but worth surfacing rather than leaving in a
        # CSV.
        if "EPE_ped" in m and m["EPE_ped"] > PED_CAUTION_RATIO * epe:
            print("  CAUTION: pedestrian-region EPE is %.3f px, %.1fx the gated figure and "
                  "%.2fx mean |GT| there.\n"
                  "           The aggregate passes on low-magnitude pixels; the pedestrian "
                  "region is considerably worse."
                  % (m["EPE_ped"], m["EPE_ped"] / epe, m["EPE_ped"] / m["gtmag_ped"]))

        mag = magnitude_table(
            os.path.join(args.carla_eval, "%s_%s_epe_vs_magnitude.csv" % (prefix, args.capture)),
            dsec_metrics_path.replace("_metrics.csv", "_epe_vs_magnitude.csv")
            if dsec_metrics_path else "")
        if mag is not None:
            print("\n  at matched |GT| magnitude (dsec_matched vs %s):"
                  % os.path.basename(dsec_metrics_path).replace("_metrics.csv", ""))
            print("    %-14s %9s %9s %8s" % ("|GT| bin (px)", "CARLA", "DSEC", "ratio"))
            for _, r in mag.iterrows():
                print("    %-14s %9.3f %9.3f %7.2fx"
                      % ("%.0f-%.0f" % (r["bin_lo"], r["bin_hi"]),
                         r["EPE_carla"], r["EPE_dsec"], r["EPE_carla"] / r["EPE_dsec"]))
        print()

    print("=" * 78)
    print("GATE")
    print("=" * 78)
    for _, model, _ in MODELS:
        if model in verdicts:
            print("  %-28s %s" % (model, "PASS" if verdicts[model] else "FAIL"))
        else:
            print("  %-28s not evaluated on CARLA yet" % model)

    if missing:
        print("\n%d of %d models have no CARLA numbers. Remediation applies to all models or "
              "none, so the decision cannot be made yet." % (len(missing), len(MODELS)))
        raise SystemExit(2)

    if all(verdicts.values()):
        print("\nAll models pass. No fine-tune needed.")
        raise SystemExit(0)

    print("\nAt least one model fails. Calibrate DVS sensor statistics before fine-tuning, and "
          "fine-tune all models or none.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
