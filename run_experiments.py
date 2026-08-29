import argparse
import csv
import os
import time

from data_prep import DATA, ensure, list_images
from experiment_consts import (ANISO, BOX, CAT_PROMPT, DISK, DOG_PROMPT, FFHQ_PROMPT,
                               GAUSS, MEDICAL_PROMPT, METFACES_PHOTO_PROMPT,
                               METFACES_PROMPT, MOTION, SNGFACES_PROMPT, SR16)
from run_batch import run_batch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "batch_results_table6")

# -- Table 6: FFHQ-1024, gauss sigma=6 / motion / SR x16, sigma_y=0.01 ----------------------
TABLE6 = [
    ("ffhq1024", "deblurring_gaussian", GAUSS, FFHQ_PROMPT, "table6_gauss"),
    ("ffhq1024", "deblurring_motion", MOTION, FFHQ_PROMPT, "table6_motion"),
    ("ffhq1024", "sr_x16", SR16, FFHQ_PROMPT, "table6_srx16"),
]

# (dataset, problem, overrides, prompt, tag)
EXPERIMENTS = [
    *TABLE6,

    # -- SNGFaces
    ("ood_sngfaces", "deblurring_gaussian", GAUSS, SNGFACES_PROMPT, "sngfaces_gauss"),
    ("ood_sngfaces", "deblurring_motion", MOTION, SNGFACES_PROMPT, "sngfaces_motion"),
    ("ood_sngfaces", "sr_x16", SR16, SNGFACES_PROMPT, "sngfaces_srx16"),
    ("ood_sngfaces", "deblurring_motion", MOTION, FFHQ_PROMPT, "sngfaces_motion_photoprompt"),

    # -- Chest X-ray
    ("ood_medical", "deblurring_gaussian", GAUSS, MEDICAL_PROMPT, "medical_gauss"),
    ("ood_medical", "deblurring_motion", MOTION, MEDICAL_PROMPT, "medical_motion"),
    ("ood_medical", "sr_x16", SR16, MEDICAL_PROMPT, "medical_srx16"),

    # -- Old experiments
    # ("ood_metfaces", "deblurring_gaussian", GAUSS, METFACES_PROMPT, "metfaces_gauss"),
    # ("ood_metfaces", "deblurring_motion", MOTION, METFACES_PROMPT, "metfaces_motion"),
    # ("ood_metfaces", "sr_x16", SR16, METFACES_PROMPT, "metfaces_srx16"),
    # ("ood_metfaces", "deblurring_motion", MOTION, METFACES_PHOTO_PROMPT, "metfaces_motion_photoprompt"),
    # ("afhq512_dog", "deblurring_gaussian", ["problem.sigma_kernel=10"], DOG_PROMPT, "table1_gauss"),
    # ("afhq512_dog", "sr_x32", [], DOG_PROMPT, "table1_srx32"),
    # ("afhq512_cat", "deblurring_gaussian", ["problem.sigma_kernel=10"], CAT_PROMPT, "table1_gauss"),
    # ("afhq512_cat", "sr_x32", [], CAT_PROMPT, "table1_srx32"),
    # ("afhq512_dog", "deblurring_gaussian", ["problem.sigma_kernel=10"], WRONG_PROMPT, "table9_wrongprompt_gauss"),
    # ("afhq512_dog", "sr_x32", [], WRONG_PROMPT, "table9_wrongprompt_srx32"),
]

# -- Morozov: The discrepancy principle
MOROZOV_OPS = {"gauss": ("deblurring_gaussian", GAUSS),
               "motion": ("deblurring_motion", MOTION),
               "srx8": ("sr_x16", SR16),
               "box": ("inpainting_squared_mask", BOX),
               "disk": ("deblurring_disk", DISK),
               "aniso": ("deblurring_aniso", ANISO)}

DELTA_RULES = {"ladder": [], "morozov": ["+problem.delta_rule=morozov"]}

# Same dataset, prompt and seed per arm.
EXPERIMENTS_MOROZOV = [
    ("ffhq1024", problem, pinned + rule_ov, FFHQ_PROMPT, f"dtune_{op}_{rule}")
    for op, (problem, pinned) in MOROZOV_OPS.items()
    for rule, rule_ov in DELTA_RULES.items()]

# -- LATINO-PRO experiments, table 6 FFHQ only
EXPERIMENTS_PRO = [(ds, prob, ov, prompt, tag.replace("table6_", "table6pro_"))
                   for ds, prob, ov, prompt, tag in TABLE6]

# Whether to use base LATINO or LATINO-PRO
IS_PRO = {e[4]: False for e in EXPERIMENTS}
IS_PRO.update({e[4]: False for e in EXPERIMENTS_MOROZOV})
IS_PRO.update({e[4]: True for e in EXPERIMENTS_PRO})
ALL_EXPERIMENTS = EXPERIMENTS + EXPERIMENTS_PRO


def main():
    ap = argparse.ArgumentParser(description="Run the LATINO experiments at native 1024^2.")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--tag", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--arm", choices=["latino", "pro", "both", "morozov"], default="latino",
                    help="which solver's arms to run (default: %(default)s)")
    ap.add_argument("--sapg-steps", type=int,
                    help="PRO arms only: override num_SAPG_steps (default 15, the paper's M)")
    ap.add_argument("--prior-samples", choices=["full", "off"], default="full",
                    help="PRO arms only; 'off' shifts the RNG stream, see run_batch.py")
    ap.add_argument("--iter-images", choices=["all", "none"], default="all",
                    help="'none' deletes each image's iter/ dir after scoring")
    args = ap.parse_args()

    pool = {"latino": EXPERIMENTS, "pro": EXPERIMENTS_PRO, "both": ALL_EXPERIMENTS,
            "morozov": EXPERIMENTS_MOROZOV}[args.arm]
    keys = set(args.only) if args.only else None
    todo = [e for e in pool if keys is None or e[0] in keys]
    if args.tag:
        todo = [e for e in todo if e[4] in args.tag]
    if not todo:
        raise SystemExit("no experiments matched")

    print(f"{len(todo)} experiment(s):")
    for ds, prob, ov, prompt, tag in todo:
        print(f"  {tag:<28} {'PRO' if IS_PRO[tag] else 'LATINO':<7} {ds:<16} {prob:<22} "
              f"prompt={prompt!r}")
    if args.dry_run:
        return

    if not args.skip_fetch:
        for ds in dict.fromkeys(e[0] for e in todo):
            ensure(ds)

    os.makedirs(RESULTS, exist_ok=True)

    cols = ["tag", "arm", "dataset", "problem", "prompt", "overrides", "n_ok",
            "PSNR_1024", "SSIM_1024", "LPIPS_1024", "FID_1024", "fid_n_images_1024",
            "PSNR_512", "SSIM_512", "LPIPS_512", "FID_512", "fid_n_images_512",
            "seconds", "returncode", "out_dir"]
    index_path = os.path.join(RESULTS, "index.csv")

    def write_index(index):
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in index:
                w.writerow(r)

    index = []
    for ds, prob, ov, prompt, tag in todo:
        n = len(list_images(ds))
        if not n:
            print(f"\nSKIP {tag} -- no images in {os.path.join(DATA, ds)}")
            continue
        out = os.path.join(RESULTS, f"{ds}__{tag}")
        if not prompt:
            raise SystemExit(f"experiment {tag!r} has no prompt; every arm must state one")
        print("\n" + "=" * 78)
        print(f"RUN {tag}  [{'LATINO-PRO' if IS_PRO[tag] else 'LATINO'}] ({ds}, {prob}, n={n})")
        t0 = time.time()
        try:
            s = run_batch(images_dir=os.path.join(DATA, ds), problem=prob, prompt=prompt,
                          out_dir=out, overrides=ov, limit=args.limit,
                          compute_fid=not args.no_fid, iter_images=args.iter_images,
                          pro=IS_PRO[tag],
                          sapg_steps=args.sapg_steps if IS_PRO[tag] else None,
                          prior_samples=args.prior_samples if IS_PRO[tag] else "full")
            rc = 0
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            s, rc = None, 1
        elapsed = time.time() - t0
        print(f"    {elapsed:.1f}s")

        row = dict(tag=tag, arm="LATINO-PRO" if IS_PRO[tag] else "LATINO", dataset=ds,
                   problem=prob, prompt=prompt, overrides=";".join(ov), out_dir=out,
                   returncode=rc, seconds=round(elapsed, 1))
        if s is not None:
            for res in ("1024", "512"):
                row.update({f"{k}_{res}": s["mean"].get(f"{k}_{res}")
                            for k in ("PSNR", "SSIM", "LPIPS")})
                row[f"FID_{res}"] = s.get(f"FID_{res}")
                row[f"fid_n_images_{res}"] = s.get(f"fid_n_images_{res}")
            row["n_ok"] = s.get("n_ok")
        index.append(row)
        write_index(index)

    print("\n" + "=" * 78)
    print(f"{sum(1 for r in index if r['returncode'] == 0)}/{len(index)} experiments ok")
    print(f"index -> {index_path}")
    for r in index:
        if r["returncode"] == 0 and r.get("PSNR_1024") is not None:
            fid = f"  FID {r['FID_1024']:.2f}" if r.get("FID_1024") is not None else ""
            line = (f"  {r['tag']:<28} {r['arm']:<11} PSNR {r['PSNR_1024']:.3f}  "
                    f"LPIPS {r['LPIPS_1024']:.4f}{fid}")
            if r.get("PSNR_512") is not None:
                line += (f"   | 512: PSNR {r['PSNR_512']:.3f}  "
                         f"LPIPS {r['LPIPS_512']:.4f}")
            print(line)


if __name__ == "__main__":
    main()
