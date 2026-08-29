""" Measures how well the SDXL VAE round-trips each dataset by doing D(E(x)) """
import os, time, json
import numpy as np, torch, torchmetrics
from PIL import Image, ImageDraw
import deepinv as dinv
from diffusers import AutoencoderKL

from data_prep import ensure

DEV = "cuda"
SIZE = 1024
DTYPE = torch.float16
N_IMAGES = 50
STRIPS_IN_COMPOSITE = 3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vae_domain_check")
os.makedirs(OUT, exist_ok=True)

start = time.time()
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=DTYPE).to(DEV).eval()
scaling_factor = vae.config.scaling_factor
lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity("vgg").to(DEV)
psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(DEV)
ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(DEV)
print(f"models ready ({time.time() - start:.0f}s)", flush=True)

downsampler = dinv.physics.Downsampling(
    img_size=(3, SIZE, SIZE), factor=2, device=DEV,
    noise_model=dinv.physics.GaussianNoise(sigma=0), filter="bicubic")
upsample = downsampler.A_adjoint  # 512 -> 1024


def load_image(path):
    """Load and preprocess exactly as the solver does: crop to a multiple of 8, upsample
    512s to 1024 if needed, then normalize to [0,1]."""
    img = Image.open(path).convert("RGB")
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.).permute(2, 0, 1)[None].to(DEV)
    h, w = x.shape[-2:]
    x = x[:, :, :h - h % 8, :w - w % 8]
    if x.shape[-2:] == (512, 512):
        x = upsample(x).clamp(0, 1)
    elif x.shape[-2:] != (SIZE, SIZE):
        raise ValueError(f"{path}: expected 512 or {SIZE} square, got {tuple(x.shape[-2:])}")
    return (x - x.min()) / (x.max() - x.min())


def create_grating_image(freq):
    """Square-wave stripes. The VAE is known to fail on these."""
    xs = np.linspace(0, 1, SIZE)[None, :].repeat(SIZE, 0)
    bars = ((np.sin(2 * np.pi * freq * xs) > 0) * 255).astype(np.uint8)
    x = torch.from_numpy(np.stack([bars] * 3, 0)[None].astype(np.float32) / 255.).to(DEV)
    return (x - x.min()) / (x.max() - x.min())


groups = {
    "ffhq             (control, native 1024)": [load_image(p) for p in ensure("ffhq1024", N_IMAGES)],
    "sngfaces         (NEAR ood, native 1024)": [load_image(p) for p in ensure("ood_sngfaces", N_IMAGES)],
    "medical          (FAR ood, native 1024)": [load_image(p) for p in ensure("ood_medical", N_IMAGES)],
    "grating          (known VAE failure)": [create_grating_image(64), create_grating_image(256)],
    # --- old:
    # "afhq_dog         (control, 512->1024)": [load_image(p) for p in ensure("afhq512_dog", N_IMAGES)],
    # "afhq_cat         (control, 512->1024)": [load_image(p) for p in ensure("afhq512_cat", N_IMAGES)],
    # "metfaces         (NEAR ood, native 1024)": [load_image(p) for p in ensure("ood_metfaces", N_IMAGES)],
}

rows, strips = [], []
for name, images in groups.items():
    for i, x in enumerate(images):
        with torch.no_grad():
            z = vae.encode((x * 2 - 1).to(DTYPE)).latent_dist.mean * scaling_factor
            restored = ((vae.decode(z / scaling_factor).sample) / 2 + 0.5).clamp(0, 1).float()
        psnr.reset()
        ssim.reset()
        lpips.reset()
        m = dict(group=name, idx=i,
                 PSNR=psnr(restored, x).item(),
                 SSIM=ssim(restored, x).item(),
                 LPIPS=lpips(restored * 2 - 1, x * 2 - 1).item())
        rows.append(m)
        print("  %-46s #%2d  PSNR %6.2f  SSIM %.4f  LPIPS %.4f  (%.0fs)"
              % (name, i, m["PSNR"], m["SSIM"], m["LPIPS"], time.time() - start), flush=True)
        error = ((restored - x).abs() * 5).clamp(0, 1)
        side_by_side = torch.cat([x, restored, error], dim=-1)[0].permute(1, 2, 0).cpu().numpy()
        strips.append((name, i, m, (side_by_side * 255).astype(np.uint8)))

# Save every strip, but keep the composite to a few rows per group.
TH = 300
tiles, count = [], {}
for name, i, m, arr in strips:
    img = Image.fromarray(arr).resize((TH * 3, TH), Image.BICUBIC)
    img.save(os.path.join(OUT, f"{name.split()[0]}_{i:02d}_gt_vs_vae.png"))
    count[name] = count.get(name, 0) + 1
    if count[name] > STRIPS_IN_COMPOSITE:
        continue
    tile = Image.new("RGB", (TH * 3, TH + 26), "white")
    tile.paste(img, (0, 26))
    ImageDraw.Draw(tile).text((4, 6),
                              f"{name}  #{i}   PSNR {m['PSNR']:.2f}   SSIM {m['SSIM']:.4f}   LPIPS {m['LPIPS']:.4f}"
                              "        [ GT | VAE round-trip D(E(x)) | 5x abs error ]", fill="black")
    tiles.append(tile)

composite = Image.new("RGB", (TH * 3, sum(t.height for t in tiles)), "white")
y = 0
for tile in tiles:
    composite.paste(tile, (0, y))
    y += tile.height
composite.save(os.path.join(OUT, "composite.png"))

print("\n%-46s %5s %16s %16s %16s"
      % ("group", "n", "PSNR mean+-sd", "SSIM mean+-sd", "LPIPS mean+-sd"))
summary = {}
for name in groups:
    vals = [r for r in rows if r["group"] == name]
    stats = {}
    for k in ("PSNR", "SSIM", "LPIPS"):
        v = np.array([r[k] for r in vals])
        stats[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        min=float(v.min()), max=float(v.max()), median=float(np.median(v)))
    summary[name] = stats
    print("%-46s %5d %8.2f+-%-6.2f %8.4f+-%-6.4f %8.4f+-%-6.4f"
          % (name, len(vals), stats["PSNR"]["mean"], stats["PSNR"]["sd"],
             stats["SSIM"]["mean"], stats["SSIM"]["sd"],
             stats["LPIPS"]["mean"], stats["LPIPS"]["sd"]))

print("\nPSNR min / median / max:")
for name in groups:
    p = summary[name]["PSNR"]
    print("  %-46s %6.2f / %6.2f / %6.2f" % (name, p["min"], p["median"], p["max"]))

json.dump({"dtype": str(DTYPE), "scaling_factor": float(scaling_factor),
           "n_images": N_IMAGES, "per_image": rows, "group_stats": summary},
          open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print(OUT)
