""" Dataset preparations """

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, list_repo_files
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MANIFESTS = os.path.join(DATA, "manifests")

IMG_EXTENTIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(name):
    d = os.path.join(DATA, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTENTIONS))


def distinct_groups(name):
    """Take distinct patients from the X-Ray dataset (each batch belong to the same patient"""
    gk = DATASETS.get(name, {}).get("group_key")
    if gk is None:
        return None
    return len({gk(os.path.splitext(os.path.basename(p))[0].split("__")[-1])
                for p in list_images(name)})


def write_manifest(name, source):
    """Filename + sha256 for each image, to persist everything."""
    os.makedirs(MANIFESTS, exist_ok=True)
    paths = list_images(name)
    out = os.path.join(MANIFESTS, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# dataset: {name}\n# source: {source}\n")
        f.write(f"# prepared: {datetime.now(timezone.utc).isoformat()}\n# n: {len(paths)}\n")
        groups = distinct_groups(name)
        if groups is not None:
            f.write(f"# distinct groups: {groups}\n")
        for p in paths:
            f.write(f"{sha256_file(p)}  {os.path.basename(p)}\n")
    return out, len(paths)


def fetch_parquet(key, spec, target_dir):
    """Read images straight out of the Hub's parquet shards.

    `split_prefix` restricts which shards are read (e.g. "data/val-" for a val split).
    Without it takes shards by appeared order."""
    repo, want, prefix = spec["repo"], spec["n"], spec.get("split_prefix")
    shards = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                    if f.endswith(".parquet") and (prefix is None or f.startswith(prefix)))
    if not shards:
        raise RuntimeError(f"{repo}: no parquet shards matching {prefix!r}")
    saved = 0
    for shard in shards:
        if saved >= want:
            break
        pf = pq.ParquetFile(hf_hub_download(repo, shard, repo_type="dataset"))
        for rg in range(pf.num_row_groups):
            if saved >= want:
                break
            table = pf.read_row_group(rg).to_pydict()
            labels = table.get(spec["label_col"]) if spec.get("label_col") else None
            for i, rec in enumerate(table["image"]):
                if saved >= want:
                    break
                if labels is not None and labels[i] != spec["label"]:
                    continue
                raw = rec["bytes"] if isinstance(rec, dict) else rec
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img.save(os.path.join(target_dir, f"{key}_{saved:04d}.png"))
                saved += 1
    if saved < want:
        print(f"  ! only {saved}/{want} available from {repo}")
    return f"hf:{repo}" + (f"::{prefix}" if prefix else "")


def nih_patient(stem):
    """NIH CXR14 names its members `<patient_id>_<followup_index>`. The patient is the part
    before the underscore, and the archive ships every follow-up study, so consecutive
    members are frequently the same person."""
    return stem.split("_")[0]


def fetch_zip(key, spec, target_dir):
    """Read images out of a Hub-hosted zip without downloading it: the index sits at the
    end of the file, so range requests can pull single members.

    `group_key` collapses a group down to one image - so we'll take 1 image per NIH patient."""
    path = f"datasets/{spec['repo']}/{spec['archive']}"
    group_key = spec.get("group_key")
    saved, seen = 0, set()
    with HfFileSystem().open(path, "rb") as fh:
        z = zipfile.ZipFile(fh)
        names = [n for n in z.namelist()
                 if n.lower().endswith(IMG_EXTENTIONS) and "__MACOSX" not in n]
        for n in names:
            if saved >= spec["n"]:
                break
            stem = os.path.splitext(os.path.basename(n))[0]
            if group_key is not None:
                g = group_key(stem)
                if g in seen:
                    continue
                seen.add(g)
            img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            img.save(os.path.join(target_dir, f"{key}_{saved:04d}__{stem}.png"))
            saved += 1
    if saved < spec["n"]:
        print(f"  ! only {saved}/{spec['n']} extracted"
              + (f" ({len(names)} members scanned)" if group_key else ""))
    return (f"hf:{spec['repo']}::{spec['archive']}"
            + (f"::one-per-{group_key.__name__}" if group_key else ""))


def fetch_files(key, spec, target_dir):
    """Download files from a Hub repo by name."""
    for i in range(spec["n"]):
        rel = spec["path"](i)
        src = hf_hub_download(spec["repo"], rel, repo_type="dataset")
        shutil.copy(src, os.path.join(target_dir, os.path.basename(rel)))
    return f"hf:{spec['repo']}"


def fetch_sngfaces(key, spec, target_dir):
    """
    Uses two static metadata files: sngfaces_metadata.json (the dataset's own per-image record) and
    sngfaces_1024_fileids.json (inventory id -> Drive file id for the 1024x1024 PNGs) to download the correct images.
    """
    with open(os.path.join(DATA, "sngfaces_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(DATA, "sngfaces_1024_fileids.json"), encoding="utf-8") as f:
        file_ids = json.load(f)

    def is_oil_portrait(inv_id):
        p = meta[inv_id]["properties"]
        technique = p.get("technique", "").lower()
        genre = p.get("genre", "").lower()
        return ("oil" in technique or "olej" in technique) and "portr" in genre

    accepted = [inv_id for inv_id in sorted(meta) if is_oil_portrait(inv_id)][:spec["n"]]

    def download(file_id):
        for confirm in (None, "t"):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            if confirm:
                url += f"&confirm={confirm}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                ctype, body = r.headers.get("Content-Type", ""), r.read()
            if "image" in ctype:
                return body
        raise RuntimeError(f"gdrive file {file_id}: expected an image, got {ctype!r}")

    def fetch_one(i, inv_id):
        raw = download(file_ids[inv_id])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        stem = inv_id.replace(":", "_")
        img.save(os.path.join(target_dir, f"{key}_{i:04d}__{stem}.png"))

    saved = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(fetch_one, i, inv_id) for i, inv_id in enumerate(accepted)]
        for fut in as_completed(futs):
            fut.result()  # surface the first worker exception instead of swallowing it
            saved += 1
    if saved < spec["n"]:
        print(f"  ! only {saved}/{spec['n']} passed the oil-portrait filter "
              f"(out of {len(meta)} SNGFaces images)")
    return "gdrive:kondela/sngfaces-dataset::processed/1024"


DATASETS = {
    "ffhq1024": dict(n=100, fetch=fetch_files,
                     repo="marcosv/ffhq-dataset",
                     path=lambda i: f"Part{i // 10000 + 1}/{i:05d}.png"),
    "ffhq512": dict(n=100, fetch=fetch_parquet,
                    repo="Ryan-sjtu/ffhq512-caption"),
    "afhq512_dog": dict(n=50, fetch=fetch_parquet,
                        repo="zzsi/afhq512_16k", split_prefix="data/val-",
                        label_col="label", label=2),  # 2 = dog
    "afhq512_cat": dict(n=50, fetch=fetch_parquet,
                        repo="zzsi/afhq512_16k", split_prefix="data/val-",
                        label_col="label", label=1),  # 1 = cat
    "ood_metfaces": dict(n=100, fetch=fetch_parquet,
                         repo="huggan/metfaces"),
    "ood_sngfaces": dict(n=100, fetch=fetch_sngfaces,
                         repo="kondela/sngfaces-dataset"),
    "ood_medical": dict(n=100, fetch=fetch_zip,
                        repo="alkzar90/NIH-Chest-X-ray-dataset",
                        archive="data/images/images_001.zip",
                        group_key=nih_patient),
}


def prepare(name, force=False):
    """Download a dataset if it isn't already on disk. Returns its image paths."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {list(DATASETS)}")
    spec = DATASETS[name]
    d = os.path.join(DATA, name)
    os.makedirs(d, exist_ok=True)

    if force:
        for p in list_images(name):
            os.remove(p)
    if len(list_images(name)) < spec["n"]:
        source = spec["fetch"](name, spec, d)
        write_manifest(name, source)
        print(f"    {name}: fetched from {source}")
    return list_images(name)


def ensure(name, n=None):
    """Image paths for a dataset, fetching first if needed, so callers do not have to run
    `prepare` beforehand."""
    want = n or DATASETS[name]["n"]
    paths = list_images(name)
    if len(paths) < want:
        paths = prepare(name)
    if len(paths) < want:
        raise RuntimeError(f"{name}: wanted {want} images, only {len(paths)} available")
    return paths[:want]


def main():
    ap = argparse.ArgumentParser(description="Fetch datasets and write manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("prepare")
    p.add_argument("--only", nargs="+")
    p.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.cmd == "list":
        print(f"{'dataset':<16} {'have':>5} {'want':>5} {'groups':>7}  {'status':<12} source")
        print("-" * 88)
        for name, spec in DATASETS.items():
            have = len(list_images(name))
            groups = distinct_groups(name)
            status = "ready" if have >= spec["n"] else ("partial" if have else "not fetched")
            print(f"{name:<16} {have:>5} {spec['n']:>5} "
                  f"{'-' if groups is None else groups:>7}  {status:<12} {spec['repo']}")
        print(f"\ndata root: {DATA}")
    else:
        for name in args.only or list(DATASETS):
            print(f"\n=== {name} ===")
            try:
                paths = prepare(name, force=args.force)
                print(f"    {len(paths)} images ready")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
