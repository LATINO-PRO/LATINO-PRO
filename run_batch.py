"""Batch runner for LATINO and LATINO-PRO.

The RNG is re-seeded before every image, so image is independent of the ones before it.
"""

import argparse
import csv
import json
import os
import random
import shutil
import time

import deepinv as dinv
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchmetrics
from diffusers import AutoencoderKL, DiffusionPipeline, LCMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_problems import get_forward_model
from noise_schemes import noise_pred_cond_y, noise_pred_cond_y_PRO
from utils import _get_x_init, crop_to_multiple, find_available_filename, get_filename_from_path, load_image_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
TIMESTEPS = [999, 874, 749, 624, 499, 374, 249, 124]
TIMESTEPS_PRO_SAPG = [999, 749, 499, 249]
FID_FEATURES = 2048
FID_WARN_BELOW = 500
SCORE_AT = 512  # the true GT for the 512-native sets, a downsample for the 1024 ones


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(problem, prompt, seed, overrides=(), pro=False, sapg_steps=None) -> DictConfig:
    overrides = list(overrides)
    if problem:
        overrides.insert(0, f"problem={problem}")
    if seed is not None:
        overrides.append(f"seed={seed}")
    if sapg_steps is not None:
        overrides.append(f"num_SAPG_steps={sapg_steps}")
    name = "LATINO-PRO" if pro else "LATINO"
    with initialize_config_dir(version_base=None, config_dir=os.path.join(HERE, "configs")):
        cfg = compose(config_name=name, overrides=overrides)
    if pro:
        if "num_SAPG_steps" not in cfg:
            raise SystemExit("configs/LATINO-PRO.yaml is missing num_SAPG_steps")
    elif cfg.model != "LATINO":
        raise SystemExit(f"only model: LATINO is supported, got {cfg.model!r}")
    if prompt:
        cfg.image.prompt = prompt
    return cfg


def collect_images(images_dir, limit=None):
    paths = sorted(os.path.join(images_dir, f) for f in os.listdir(images_dir)
                   if f.lower().endswith(IMG_EXTS))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images in {images_dir}")
    return paths


def reset_scheduler_timesteps(pipe, device, steps=8):
    """Reset the scheduler before every image"""
    schedule = {8: TIMESTEPS, 4: TIMESTEPS_PRO_SAPG}[steps]
    pipe.scheduler.set_timesteps(steps, device=device)
    pipe.scheduler.timesteps = torch.tensor(schedule, device=device, dtype=torch.long)


def project_onto_ball(theta, theta_0, radius):
    """Project theta onto a ball of radius `radius` centered at `theta_0`."""
    delta = theta - theta_0
    norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True)
    scaling_factor = torch.clamp(radius / (norm_delta + 1e-8), max=1.0)
    if (scaling_factor < 1.0).any():
        print("Projected!")
    return theta_0 + scaling_factor * delta


def build_pipeline(cfg, device):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config(base, subfolder="unet")).to(device, torch.float16)
    unet.load_state_dict(torch.load(
        hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_4step_unet_fp16.bin"),
        map_location=device, weights_only=True))
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(base, unet=unet, vae=vae,
                                             torch_dtype=torch.float16, variant="fp16",
                                             guidance_scale=0).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    reset_scheduler_timesteps(pipe, device)
    return pipe


def build_conditioning(pipe, cfg, device):
    text_embeddings, _, pooled, _ = pipe.encode_prompt(
        cfg.image.prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=False)
    time_ids = pipe._get_add_time_ids(
        original_size=(1024, 1024), crops_coords_top_left=(0, 0), target_size=(1024, 1024),
        dtype=torch.float16, text_encoder_projection_dim=1280).to(device)
    return text_embeddings, {"text_embeds": pooled, "time_ids": time_ids}


def build_metrics(device):
    return (torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity("vgg").to(device),
            torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device),
            torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device))


def make_fid(device):
    """One accumulator per resolution."""
    try:
        return {res: FrechetInceptionDistance(feature=FID_FEATURES, normalize=True).to(device)
                for res in ("512", "1024")}
    except ModuleNotFoundError as e:
        raise SystemExit(f"batch FID needs torch-fidelity:\n"
                         f"    pip install torch-fidelity==0.3.0\n"
                         f"or pass --no-fid.\n({e})")


def _downsample_to_score_res(x, device):
    """1024 -> 512 with the same bicubic Downsampling main_LATINO.py uses elsewhere."""
    assert x.shape[-1] % SCORE_AT == 0, f"width {x.shape[-1]} isn't a multiple of {SCORE_AT}"
    factor = x.shape[-1] // SCORE_AT
    if factor <= 1:
        return x
    op = dinv.physics.Downsampling(img_size=x.shape[1:], factor=factor, device=device,
                                   noise_model=dinv.physics.GaussianNoise(sigma=0),
                                   filter="bicubic")
    return op.A(x.float()).clamp(0, 1)


def prepare_inputs(cfg, device):
    """Ground truth, forward operator and measurement."""
    raw = crop_to_multiple(load_image_tensor(cfg.image.path), m=8).to(device)
    true_clean_512 = None
    if raw.shape[-1] == 512:
        # Keep the true 512^2 GT for scoring, separate from the 1024^2-promoted version,
        # because A(A_adjoint(x)) is lossy
        true_clean_512 = (raw - raw.min()) / (raw.max() - raw.min())
        clean = dinv.physics.Downsampling(
            img_size=(3, 1024, 1024), factor=2, device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=0),
            filter="bicubic").A_adjoint(raw).clamp(0, 1)
    else:
        clean = raw
    clean = (clean - clean.min()) / (clean.max() - clean.min())

    forward_model, transpose_operator = get_forward_model(cfg, clean, device)
    y = forward_model(clean)
    y_norm = y * 2 - 1
    sigma_y_norm = cfg.problem.sigma_y * 2
    return clean, true_clean_512, forward_model, transpose_operator, y_norm, sigma_y_norm


def score_and_save(restored, clean, true_clean_512, y_norm, out_dir, device, metrics, fid,
                   elapsed):
    """Save the scored PNGs, compute the metrics at both resolutions, update FID."""
    save_image(restored, os.path.join(out_dir, "restored_1024.png"))
    save_image(((y_norm + 1) / 2).clamp(0, 1).detach().cpu(), os.path.join(out_dir, "degraded.png"))
    save_image(clean.detach().cpu(), os.path.join(out_dir, "clean_1024.png"))

    restored_s = _downsample_to_score_res(restored, device)
    # 512^2-native datasets score against the true original; 1024^2-native are downsampled to score at 512^2.
    clean_s = true_clean_512 if true_clean_512 is not None else _downsample_to_score_res(clean, device)
    save_image(restored_s, os.path.join(out_dir, "restored.png"))
    save_image(clean_s.detach().cpu(), os.path.join(out_dir, "clean.png"))

    lpips, psnr, ssim = metrics
    result = {}
    for tag, (r, c) in {"1024": (restored, clean), "512": (restored_s, clean_s)}.items():
        lpips.reset()
        psnr.reset()
        ssim.reset()
        result[f"PSNR_{tag}"] = psnr(r, c).item()
        result[f"SSIM_{tag}"] = ssim(r, c).item()
        result[f"LPIPS_{tag}"] = lpips(r * 2 - 1, c * 2 - 1).item()
    print(", ".join(f"{k}: {v:.3f}" for k, v in result.items()))
    with open(os.path.join(out_dir, "metrics.csv"), "w+") as f:
        f.write(json.dumps(result))

    if fid is not None:
        with torch.no_grad():
            fid["1024"].update(clean.float().clamp(0, 1), real=True)
            fid["1024"].update(restored.float().clamp(0, 1), real=False)
            fid["512"].update(clean_s.float().clamp(0, 1), real=True)
            fid["512"].update(restored_s.float().clamp(0, 1), real=False)

    result["seconds"] = elapsed
    return result


def reconstruct_one(pipe, cfg, device, text_embeddings, added_cond_kwargs, out_dir, metrics, fid):
    clean, true_clean_512, forward_model, transpose_operator, y_norm, sigma_y_norm = \
        prepare_inputs(cfg, device)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w+") as f:
        OmegaConf.save(config=cfg, f=f)

    mask = forward_model.mask if cfg.problem.type == "inpainting_squared_mask" else None
    x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
    with torch.no_grad():
        qz = pipe.vae.encode(x_init.clip(-1, 1).half())
    mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor
    if cfg.init_strategy == "y_noise":
        latents = pipe.scheduler.add_noise(mu_z, noise=torch.randn_like(mu_z),
                                           timesteps=torch.tensor([999]))
    else:
        latents = mu_z

    reset_scheduler_timesteps(pipe, device)
    started = time.time()
    for step, timestep in enumerate(pipe.scheduler.timesteps):
        text_embeddings = text_embeddings.detach().requires_grad_(True)
        with torch.no_grad():
            noise_uncond = pipe.unet(latents, timestep, encoder_hidden_states=text_embeddings,
                                     added_cond_kwargs=added_cond_kwargs).sample
        with torch.no_grad():
            _, noise_pred = noise_pred_cond_y(
                latents=latents, t=timestep, pipe=pipe, cfg=cfg, logdir=out_dir,
                y_guidance=y_norm, forward_model=forward_model,
                noise_pred=noise_uncond, sigma_y=sigma_y_norm)
        latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

    with torch.no_grad():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    restored = (decoded / 2 + 0.5).clamp(0, 1)
    elapsed = time.time() - started

    return score_and_save(restored, clean, true_clean_512, y_norm, out_dir, device,
                          metrics, fid, elapsed)


def _save_sapg_plots(out_dir):
    data = pd.read_csv(os.path.join(out_dir, "metrics_log.csv"))
    for col, color in (("PSNR", "b"), ("SSIM", "g"), ("LPIPS", "r")):
        plt.figure(figsize=(6, 5))
        plt.plot(data["Iteration"], data[col], label=col, color=color, linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel(col)
        plt.title(f"{col} Over Iterations")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_plot.png"), dpi=300)
        plt.close()


def reconstruct_one_pro(pipe, cfg, device, text_embeddings_0, added_cond_kwargs, out_dir,
                        metrics, fid, prior_samples="full", plots=False):
    # text_embeddings is the SAPG optimisation variable, so a fresh clone per image is what keeps images independent.
    text_embeddings = text_embeddings_0.detach().clone()
    accumulated_grad = torch.zeros_like(text_embeddings)

    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    latents = pipe.prepare_latents(
        batch_size=1, num_channels_latents=pipe.unet.config.in_channels,
        height=1024, width=1024, dtype=torch.float16, device=device, generator=generator)

    num_inference_steps = 4
    reset_scheduler_timesteps(pipe, device, num_inference_steps)

    clean, true_clean_512, forward_model, transpose_operator, y_norm, sigma_y_norm = prepare_inputs(cfg, device)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w+") as f:
        OmegaConf.save(config=cfg, f=f)

    mask = forward_model.mask if cfg.problem.type == "inpainting_squared_mask" else None
    x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
    save_image(x_init * 0.5 + 0.5, os.path.join(out_dir, "x_init.png"))
    with torch.no_grad():
        qz = pipe.vae.encode(x_init.clip(-1, 1).half())
    mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor
    if cfg.init_strategy == "y_noise":
        latents = pipe.scheduler.add_noise(mu_z, noise=torch.randn_like(mu_z),
                                           timesteps=torch.tensor([999]))
    elif cfg.init_strategy == "y":
        latents = mu_z
    else:
        raise SystemExit(f"unknown init_strategy {cfg.init_strategy!r}")

    csv_file = os.path.join(out_dir, "metrics_log.csv")
    with open(csv_file, mode="a", newline="") as f:
        csv.writer(f).writerow(["Iteration", "PSNR", "SSIM", "LPIPS"])

    lpips_loss, psnr_loss, ssim_loss = metrics

    def sample_prior(j):
        prior_latents = latents2.clone()
        reset_scheduler_timesteps(pipe, device, 4)
        for timestep in pipe.scheduler.timesteps:
            noise_uncond = pipe.unet(prior_latents, timestep,
                                     encoder_hidden_states=text_embeddings,
                                     added_cond_kwargs=added_cond_kwargs).sample
            prior_latents = pipe.scheduler.step(noise_uncond, timestep, prior_latents).prev_sample
        decoded = pipe.vae.decode(prior_latents / pipe.vae.config.scaling_factor).sample
        save_image(torch.clamp(decoded * 0.5 + 0.5, 0, 1),
                   os.path.join(out_dir, f"prior_{j}.png"))

    started = time.time()
    for j in range(cfg.num_SAPG_steps):
        print(f"SAPG step: {j + 1}")
        for i, timestep in enumerate(pipe.scheduler.timesteps):
            print(f"Step {i + 1}: Timestep {timestep}")
            text_embeddings = text_embeddings.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                noise_uncond = pipe.unet(latents, timestep,
                                         encoder_hidden_states=text_embeddings,
                                         added_cond_kwargs=added_cond_kwargs).sample

            with torch.no_grad():
                _, noise_pred = noise_pred_cond_y_PRO(
                    latents=latents, t=timestep, pipe=pipe, cfg=cfg, logdir=out_dir,
                    y_guidance=y_norm, forward_model=forward_model,
                    noise_pred=noise_uncond, sigma_y=sigma_y_norm,
                    SAPG_j=j, n_steps=num_inference_steps)

            if i < num_inference_steps - 1:
                alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                with torch.enable_grad():
                    z0_pred_c = torch.sqrt(1 / alpha_t) * (
                            latents - torch.sqrt(1 - alpha_t) * noise_uncond)

                with torch.no_grad():
                    latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

                with torch.enable_grad():
                    alpha_t = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[i + 1]]
                    loss = -0.5 / (1 - alpha_t) * torch.norm(
                        latents - torch.sqrt(alpha_t) * z0_pred_c) ** 2
                    gradients = torch.autograd.grad(
                        loss, inputs=text_embeddings, retain_graph=False)[0]

                accumulated_grad += gradients / torch.norm(gradients, dim=-1, keepdim=True)
                accumulated_grad[0, :4] = 0
                accumulated_grad = accumulated_grad.detach()
            else:
                text_embeddings = project_onto_ball(
                    text_embeddings
                    + 0.08 * (max(0.9 ** (max(0, j - 10)), 0.001)) * accumulated_grad,
                    text_embeddings_0, 15)

                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                with torch.no_grad():
                    decoded_image = pipe.vae.decode(
                        latents / pipe.vae.config.scaling_factor).sample.clip(-1, 1)
                    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)

                lpips_loss.reset()
                psnr_loss.reset()
                ssim_loss.reset()
                psnr = psnr_loss(restored_x, clean).item()
                ssim = ssim_loss(restored_x, clean).item()
                lpips = lpips_loss(restored_x * 2 - 1, clean * 2 - 1).item()
                trace = {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips}

                psnr_loss.reset()
                restored_x_lr = forward_model.A(restored_x.float())
                trace["OBS-PSNR"] = psnr_loss(((y_norm + 1) / 2).clamp(0, 1), restored_x_lr).item()
                print(", ".join(f"{k}: {v:.3f}" for k, v in trace.items()))

                with open(csv_file, mode="a", newline="") as f:
                    csv.writer(f).writerow([j + 1, psnr, ssim, lpips])

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents2 = pipe.scheduler.add_noise(latents.detach(), noise=noise,
                                                timesteps=torch.tensor([999]))
            latents = latents2.clone()

            if prior_samples == "full":
                sample_prior(j)

            latents = latents2.clone()
        del latents2

        if j < cfg.num_SAPG_steps - 1:
            reset_scheduler_timesteps(pipe, device, num_inference_steps)
        else:
            reset_scheduler_timesteps(pipe, device, 8)
            for timestep in pipe.scheduler.timesteps:
                with torch.no_grad():
                    noise_uncond = pipe.unet(latents, timestep,
                                             encoder_hidden_states=text_embeddings,
                                             added_cond_kwargs=added_cond_kwargs).sample
                with torch.no_grad():
                    _, noise_pred = noise_pred_cond_y_PRO(
                        latents=latents, t=timestep, pipe=pipe, cfg=cfg, logdir=out_dir,
                        y_guidance=y_norm, forward_model=forward_model,
                        noise_pred=noise_uncond, sigma_y=sigma_y_norm,
                        SAPG_j=j, n_steps=8)
                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

    elapsed = time.time() - started

    if plots:
        _save_sapg_plots(out_dir)

    with torch.no_grad():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    restored = (decoded / 2 + 0.5).clamp(0, 1)

    return score_and_save(restored, clean, true_clean_512, y_norm, out_dir, device,
                          metrics, fid, elapsed)


def summarise(rows, out_root, cfg, fid, wall_seconds, run_mode=None):
    ok = [r for r in rows if "error" not in r]
    keys = [k for k in ("PSNR_1024", "SSIM_1024", "LPIPS_1024", "PSNR_512", "SSIM_512",
                        "LPIPS_512", "seconds") if any(k in r for r in ok)]
    n_per_key = {k: sum(1 for r in ok if k in r) for k in keys}
    for k, n in n_per_key.items():
        if k != "seconds" and ok and n < len(ok):
            print(f"WARNING: {k} present in only {n}/{len(ok)} ok rows -- "
                  f"mean/std below is NOT over the full set")
    summary = {
        "n_images": len(rows), "n_ok": len(ok), "n_failed": len(rows) - len(ok),
        "wall_seconds": wall_seconds,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "seed_base": cfg.seed,
        "seed_scheme": "per-image: seed_base + index",
        "n_per_key": n_per_key,
        "mean": {k: float(np.mean([r[k] for r in ok if k in r])) for k in keys} if ok else {},
        "std": ({k: float(np.std([r[k] for r in ok if k in r], ddof=1)) for k in keys}
                if len(ok) > 1 else {}),
    }
    if run_mode:
        summary["run_mode"] = run_mode
    if fid is not None:
        for tag in ("1024", "512"):
            n = int(fid[tag].real_features_num_samples)
            summary[f"fid_n_images_{tag}"] = n
            if n < 2:
                summary[f"FID_{tag}"] = None
            else:
                with torch.no_grad():
                    summary[f"FID_{tag}"] = float(fid[tag].compute().item())
                if n < FID_WARN_BELOW:
                    summary[f"fid_warning_{tag}"] = (
                        f"FID_{tag} over only {n} images; paper uses ~1k. "
                        f"Compare arms at the same N, not against published tables.")

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_root, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "path", "seed"] + keys + ["error"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n" + "=" * 70)
    print(f"done: {len(ok)}/{len(rows)} ok in {wall_seconds:.1f}s")
    for k in keys:
        if k in summary["mean"]:
            print(f"  {k:<9} {summary['mean'][k]:.4f} +- {summary['std'].get(k, 0):.4f}")
    for tag in ("1024", "512"):
        if summary.get(f"FID_{tag}") is not None:
            print(f"  {'FID_' + tag:<9} {summary[f'FID_{tag}']:.4f}   "
                  f"(over {summary[f'fid_n_images_{tag}']} images)")
    print(f"  -> {out_root}")
    return summary


def build_parser():
    ap = argparse.ArgumentParser(
        description="Batch LATINO / LATINO-PRO run.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42, help="base seed; image i runs at seed+i")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--pro", action="store_true",
                    help="run LATINO-PRO instead of LATINO; ~16x the UNet calls per image")
    ap.add_argument("--sapg-steps", type=int,
                    help="PRO only: override num_SAPG_steps (default 15, the paper's M)")
    ap.add_argument("--prior-samples", choices=["full", "off"], default="full",
                    help="PRO only. 'off' skips the prior sample, 60 of the 128 UNet "
                         "calls; it also shifts the RNG, so the two modes are not "
                         "comparable")
    ap.add_argument("--iter-images", choices=["all", "none"], default="all",
                    help="'none' deletes each image's iter/ dir after scoring; PRO writes "
                         "~200MB/image of these at 1024^2")
    ap.add_argument("--plots", action="store_true",
                    help="PRO only: write the per-image SAPG convergence plots")
    return ap


def run_batch(images_dir, problem, prompt, out_dir, overrides=(), limit=None, seed=42,
              pro=False, sapg_steps=None, prior_samples="full", iter_images="all",
              plots=False, compute_fid=True):
    """Run one arm (N images) through LATINO or LATINO-PRO, and return a summary dict.

    The pipeline is built here, so consecutive calls are independent of each other in
    everything but CUDA allocator state.
    """
    if not pro:
        misused = []
        if sapg_steps is not None:
            misused.append("--sapg-steps")
        if prior_samples != "full":
            misused.append("--prior-samples")
        if plots:
            misused.append("--plots")
        if misused:
            raise SystemExit(f"{', '.join(misused)} only apply to --pro")
    cfg = load_config(problem, prompt, seed, overrides, pro, sapg_steps)
    device = torch.device("cuda")
    paths = collect_images(images_dir, limit)

    out_root = out_dir
    os.makedirs(out_root, exist_ok=True)
    arm = "LATINO-PRO" if pro else "LATINO"
    extra = f", num_SAPG_steps={cfg.num_SAPG_steps}, prior_samples={prior_samples}" \
        if pro else ""
    print(f"{len(paths)} images -> {out_root}  ({arm}, scoring at both 1024^2 and {SCORE_AT}^2, "
          f"prompt={cfg.image.prompt!r}, problem={cfg.problem.type}, "
          f"sigma_kernel={cfg.problem.get('sigma_kernel')}, "
          f"downscaling_factor={cfg.problem.get('downscaling_factor')}{extra})")

    fid = make_fid(device) if compute_fid else None
    set_seeds(cfg.seed)
    pipe = build_pipeline(cfg, device)
    text_embeddings, added_cond_kwargs = build_conditioning(pipe, cfg, device)
    metrics = build_metrics(device)

    rows, started = [], time.time()
    base_seed = cfg.seed
    for i, path in enumerate(paths):
        name = get_filename_from_path(path)
        cfg.image.path = path
        # index-derived: stable under --limit, not under dropping images into the dir
        # later, which is why it also goes in the csv
        image_seed = base_seed + i
        cfg.seed = image_seed  # so each config.yaml re-runs standalone

        out_dir_i = os.path.join(out_root, find_available_filename(out_root, name))
        print(f"\n[{i + 1}/{len(paths)}] {path} -> {out_dir_i}  (seed {image_seed})")
        set_seeds(image_seed)
        try:
            if pro:
                result = reconstruct_one_pro(pipe, cfg, device, text_embeddings,
                                             added_cond_kwargs, out_dir_i, metrics, fid,
                                             prior_samples=prior_samples, plots=plots)
            else:
                result = reconstruct_one(pipe, cfg, device, text_embeddings, added_cond_kwargs,
                                         out_dir_i, metrics, fid)
            rows.append({"image": name, "path": path, "seed": image_seed, **result})
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append({"image": name, "path": path, "seed": image_seed,
                         "error": f"{type(e).__name__}: {e}"})

        if iter_images == "none":
            shutil.rmtree(os.path.join(out_dir_i, "iter"), ignore_errors=True)

    cfg.seed = base_seed
    run_mode = {"arm": arm}
    if pro:
        run_mode.update(num_SAPG_steps=cfg.num_SAPG_steps, prior_samples=prior_samples)
    return summarise(rows, out_root, cfg, fid, time.time() - started, run_mode=run_mode)


def main():
    args = build_parser().parse_args()
    run_batch(images_dir=args.images_dir, problem=args.problem, prompt=args.prompt,
              out_dir=args.out_dir, overrides=args.override, limit=args.limit, seed=args.seed,
              pro=args.pro, sapg_steps=args.sapg_steps, prior_samples=args.prior_samples,
              iter_images=args.iter_images, plots=args.plots, compute_fid=not args.no_fid)


if __name__ == "__main__":
    main()
