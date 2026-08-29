"""Sweep the data-fidelity weight delta and measure the effect, paired per image.

Every arm runs at the same n. Images past N_SELECT get their own columns, so a
schedule picked on the first 25 can be checked against the rest.
"""

import argparse
import csv
import math
import os
import statistics as st
import time
from collections import OrderedDict

from data_prep import DATA, ensure
from experiment_consts import (FFHQ_PROMPT, GAUSS, MEDICAL_PROMPT, METFACES_PROMPT,
                               MOTION, SNGFACES_PROMPT, SR16)
from run_batch import run_batch

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET = "LPIPS_1024"
METRICS = ["LPIPS_1024", "PSNR_1024", "SSIM_1024", "LPIPS_512", "PSNR_512"]
BASELINE = "c1"
N_SELECT = 25

# key -> (what it does, overrides).
EXPERIMENTS = {
    "c1": ("released schedule",
           ["+problem.delta_scale=1"]),
    "c0p5": ("half data weight, every step",
             ["+problem.delta_scale=0.5"]),
    "c2": ("2x data weight, every step",
           ["+problem.delta_scale=2"]),
    "c4": ("4x data weight, every step",
           ["+problem.delta_scale=4"]),
    "c8": ("8x data weight, every step",
           ["+problem.delta_scale=8"]),
    "c16": ("16x data weight, every step",
            ["+problem.delta_scale=16"]),
    "e1_l4": ("4x on the last 3 steps",
              ["+problem.delta_scale_early=1", "+problem.delta_scale_late=4"]),
    "e1_l8": ("8x on the last 3 steps",
              ["+problem.delta_scale_early=1", "+problem.delta_scale_late=8"]),
    "e1_l16": ("16x on the last 3 steps",
               ["+problem.delta_scale_early=1", "+problem.delta_scale_late=16"]),
    "e1_l32": ("32x on the last 3 steps",
               ["+problem.delta_scale_early=1", "+problem.delta_scale_late=32"]),
    "e1_l64": ("64x on the last 3 steps",
               ["+problem.delta_scale_early=1", "+problem.delta_scale_late=64"]),
    "e1_l256": ("256x on the last 3 steps",
                ["+problem.delta_scale_early=1", "+problem.delta_scale_late=256"]),
    "e1_l16_s300": ("16x on the last 2 steps",
                    ["+problem.delta_scale_early=1", "+problem.delta_scale_late=16",
                     "+problem.delta_split_t=300"]),
    "e1_l16_s500": ("16x on the last 4 steps",
                    ["+problem.delta_scale_early=1", "+problem.delta_scale_late=16",
                     "+problem.delta_split_t=500"]),
    "e1p09_l1": ("1.09x early only (gamma-matched)",
                 ["+problem.delta_scale_early=1.09", "+problem.delta_scale_late=1"]),
    "e1p47_l1": ("1.47x early only (gamma-matched)",
                 ["+problem.delta_scale_early=1.47", "+problem.delta_scale_late=1"]),
    "e1p47_l16": ("1.47x early + 16x on last 3 steps",
                  ["+problem.delta_scale_early=1.47", "+problem.delta_scale_late=16"]),
    "c1_vp0": ("drop (1-alpha_t), weight flat in t",
               ["+problem.delta_scale=1", "+problem.delta_var_power=0"]),
}

PROMPTS = {"ood_sngfaces": SNGFACES_PROMPT, "ffhq1024": FFHQ_PROMPT,
           "ood_medical": MEDICAL_PROMPT, "ood_metfaces": METFACES_PROMPT}
DEGRADATIONS = {"deblurring_gaussian": GAUSS, "deblurring_motion": MOTION, "sr_x16": SR16}


def arm_dir(problem, arm):
    return f"{problem}__{arm}"


def read_per_image(path):
    """-> OrderedDict image -> {metric: float}, in the order run_batch.py ran."""
    csv_path = os.path.join(path, "per_image.csv")
    if not os.path.isfile(csv_path):
        return None
    out = OrderedDict()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            out[r["image"]] = {k: float(r[k]) for k in METRICS if r.get(k) not in ("", None)}
    return out or None


def mean_sd(rows, key):
    if rows is None:
        return None
    vals = [v[key] for v in rows.values() if key in v]
    if not vals:
        return None
    return {"n": len(vals), "mean": st.mean(vals), "sd": st.pstdev(vals)}


def metrics_diffs(base, arm, key, images=None):
    """Paired metrics difference (arm - base) between experiment and baseline."""
    if base is None or arm is None:
        return None
    order = images if images is not None else list(base)
    diffs = [arm[im][key] - base[im][key] for im in order
             if im in base and im in arm and key in base[im] and key in arm[im]]
    if len(diffs) < 2:
        return None
    n = len(diffs)
    return {"n": n, "mean": st.mean(diffs), "sem": st.stdev(diffs) / math.sqrt(n)}


def report_results(out_root, problem, arms):
    found = OrderedDict()
    for a in arms:
        rows = read_per_image(os.path.join(out_root, arm_dir(problem, a)))
        if rows is not None:
            found[a] = rows
    if not found:
        raise SystemExit(f"no finished arms under {out_root}")

    base = found.get(BASELINE)
    images = list(next(iter(found.values())))
    sel, held = images[:N_SELECT], images[N_SELECT:]

    head = f"\n{problem}  {len(images)} images"
    if base is not None:
        head += (f", baseline {BASELINE}: {TARGET} {mean_sd(base, TARGET)['mean']:.4f}, "
                 f"PSNR {mean_sd(base, 'PSNR_1024')['mean']:.3f}")
    print(head)
    if base is None:
        print(f"  no {BASELINE} arm here, so there are no paired columns")
    cols = (f"  {'arm':<14}{'what it does':<38}{'n':>4}{TARGET:>17}{'PSNR':>9}"
            f"{'dLPIPS':>18}{'dPSNR':>9}")
    if held:
        cols += f"{'dLPIPS held':>18}{'dPSNR held':>12}"
    print(cols)

    csv_rows = []
    for arm, rows in found.items():
        label = EXPERIMENTS[arm][0]
        is_base = arm == BASELINE
        m = mean_sd(rows, TARGET)
        d = None if is_base else metrics_diffs(base, rows, TARGET, sel)
        dp = None if is_base else metrics_diffs(base, rows, "PSNR_1024", sel)
        dh = None if is_base else metrics_diffs(base, rows, TARGET, held)
        dph = None if is_base else metrics_diffs(base, rows, "PSNR_1024", held)

        psnr = mean_sd(rows, "PSNR_1024")
        line = (f"  {arm:<14}{label:<38}{m['n']:>4}{m['mean']:>9.4f}+-{m['sd']:<6.4f}"
                f"{psnr['mean']:>9.3f}")
        line += f"{d['mean']:>+10.4f}+-{d['sem']:<6.4f}" if d else f"{'--':>18}"
        line += f"{dp['mean']:>+9.3f}" if dp else f"{'--':>9}"
        if held:
            line += f"{dh['mean']:>+10.4f}+-{dh['sem']:<6.4f}" if dh else f"{'--':>18}"
            line += f"{dph['mean']:>+12.3f}" if dph else f"{'--':>12}"
        print(line + ("   (baseline)" if is_base else ""))

        row = {"problem": problem, "arm": arm, "what_it_does": label, "n": m["n"]}
        for k in METRICS:
            s = mean_sd(rows, k)
            if s:
                row[f"{k}_mean"], row[f"{k}_sd"] = s["mean"], s["sd"]
        for tag, stat in (("paired", d), ("paired_psnr", dp),
                          ("heldout", dh), ("heldout_psnr", dph)):
            if stat:
                row[f"{tag}_delta"], row[f"{tag}_sem"] = stat["mean"], stat["sem"]
                row[f"{tag}_n"] = stat["n"]
        csv_rows.append(row)

    best = min((mean_sd(r, TARGET)["mean"], a) for a, r in found.items())
    print(f"  best {TARGET}: {best[1]} at {best[0]:.4f}")
    if held:
        print(f"  held-out columns cover images {N_SELECT}..{len(images) - 1}")

    path = os.path.join(out_root, "sweep.csv")
    cols = list(OrderedDict((k, None) for r in csv_rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n-> {path}")


def main():
    ap = argparse.ArgumentParser(description="Sweep delta per timestep segment.")
    ap.add_argument("-n", "--limit", type=int, default=25, help="images per arm")
    ap.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS),
                    default=list(EXPERIMENTS), metavar="NAME")
    ap.add_argument("--dataset", default="ood_sngfaces")
    ap.add_argument("--problem", default="deblurring_motion")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "delta_sweep"))
    ap.add_argument("--fid", action="store_true", help="badly biased below a few hundred")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms, limit = args.experiments, args.limit
    out_root = os.path.abspath(args.out_dir)
    todo = [a for a in arms
            if read_per_image(os.path.join(out_root, arm_dir(args.problem, a))) is None]

    if todo:
        if args.problem not in DEGRADATIONS:
            raise SystemExit(f"no pinned overrides for {args.problem}; add them in "
                             f"run_experiments.py so the sweep matches the arms it gets "
                             f"compared against")
        if args.dataset not in PROMPTS:
            raise SystemExit(f"no prompt for dataset {args.dataset!r}")
        pinned = DEGRADATIONS[args.problem]
        prompt = PROMPTS[args.dataset]

        ensure(args.dataset)
        images_dir = os.path.join(DATA, args.dataset)

        print(f"delta sweep: {len(todo)}/{len(arms)} arms x {limit} images, "
              f"target {TARGET}")
        print(f"  images   {images_dir}")
        print(f"  prompt   {prompt!r}")
        print(f"  out      {out_root}")
        for a in arms:
            mark = "" if a in todo else "  [done, reusing]"
            print(f"    {arm_dir(args.problem, a):<40} " + " ".join(EXPERIMENTS[a][1]) + mark)
        if args.dry_run:
            return

        os.makedirs(out_root, exist_ok=True)
        started, failures = time.time(), []
        for i, arm in enumerate(todo, 1):
            name = arm_dir(args.problem, arm)
            print("\n" + "=" * 78)
            print(f"ARM {i}/{len(todo)}  {name}  {EXPERIMENTS[arm][0]}")
            t0 = time.time()
            try:
                run_batch(images_dir=images_dir, problem=args.problem, prompt=prompt,
                          out_dir=os.path.join(out_root, name),
                          overrides=list(pinned) + EXPERIMENTS[arm][1],
                          limit=limit, compute_fid=args.fid, iter_images="none")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                failures.append(name)
            print(f"    {time.time() - t0:.1f}s")

        print("\n" + "=" * 78)
        print(f"{len(todo) - len(failures)}/{len(todo)} arms ok in "
              f"{(time.time() - started) / 60:.1f} min")
        if failures:
            print("FAILED: " + ", ".join(failures))
    else:
        print(f"all {len(arms)} arms already have results under {out_root}")

    report_results(out_root, args.problem, arms)


if __name__ == "__main__":
    main()
