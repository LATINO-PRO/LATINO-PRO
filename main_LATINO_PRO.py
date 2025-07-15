import os
import json
import time
import deepinv as dinv
from torchvision.utils import save_image
import random
import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
import hydra

from utils import (
    load_image_tensor,
    crop_to_multiple,
    get_filename_from_path,
    find_available_filename,
    _get_x_init,
)
from inverse_problems import get_forward_model
from noise_schemes import (
    noise_pred_cond_y_PRO,
)

@hydra.main(version_base=None, config_path="configs", config_name="LATINO-PRO")
def main(cfg: DictConfig) -> None:
    # Set global random seeds for full reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior for CUDA (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the device
    device = torch.device("cuda")  # Use torch.device instead of string

    lpips_loss = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device)

    # load stable diffusion
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

    # Load model.
    unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device, weights_only=True))
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16", guidance_scale=0).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    prompt = cfg.image.prompt    #Ensure that the prompt starts with "a photo of"!

    # Encode text to conditioning
    text_embeddings, _, pooled_text_embeds, _ = pipe.encode_prompt(
        prompt,
        device=device, 
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )

    # Create a random generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Assuming desired resolution of 1024x1024
    image_height = 1024
    image_width = 1024

    # Prepare initial noise latents with correct device type
    latents = pipe.prepare_latents(
        batch_size=1,  # number of images to generate
        num_channels_latents=pipe.unet.config.in_channels,  # latent channels
        height=image_height,  # image height
        width=image_width,   # image width
        dtype=torch.float16,  # datatype
        device=device,  # Corrected device type
        generator=generator  # Random number generator
    )

    # Get time_ids automatically based on the image resolution
    time_ids = pipe._get_add_time_ids(
        original_size=(image_height, image_width),  # The original image resolution
        crops_coords_top_left=(0, 0),  # No cropping
        target_size=(image_height, image_width),  # Target resolution
        dtype=torch.float16,  # Ensure correct data type
        text_encoder_projection_dim=1280
    ).to(device)
    
    # Additional conditioning required for SDXL
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds,  # Pass the pooled text embeddings
        "time_ids": time_ids
    }

    # Define the number of inference steps and set timesteps
    num_inference_steps = 4     # set to 8 to use the full LATINO schedule
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Override the scheduler's timesteps with DMD2 values (or whole 8-steps)
    #custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
    custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
    pipe.scheduler.timesteps = custom_timesteps

    # load clean image
    xtemp = load_image_tensor(cfg.image.path)

    x_clean = crop_to_multiple(xtemp, m=8).to(device)
    
    # To adapt the method to 512x512 images
    if xtemp.shape[-1] == 512:
        noise_model_512_to_1024 = dinv.physics.GaussianNoise(sigma=0)

        model_512_to_1024 = dinv.physics.Downsampling(
                img_size=(3, 1024, 1024),
                factor=2,
                device=device,
                noise_model=noise_model_512_to_1024,
                filter = "bicubic"
                ).A_adjoint
    
        x_clean = model_512_to_1024(x_clean).clamp(0,1)

    x_clean = (x_clean - x_clean.min())/(x_clean.max() - x_clean.min())

    # Build forward and transpose operators
    forward_model, transpose_operator = get_forward_model(cfg, x_clean, device)
    
    y = forward_model(x_clean)
    y_norm = y * 2 - 1
    sigma_y_norm = cfg.problem.sigma_y * 2

    # create log folder
    logdir = os.curdir

    xp_log_dir = os.path.join(logdir, "results_LATINO_PRO", cfg.problem.type, cfg.log_subfolder)

    os.makedirs(xp_log_dir, exist_ok=True)
    imname = get_filename_from_path(cfg.image.path)
    xpname = find_available_filename(folder=xp_log_dir, prefix=f'{imname}')
    xp_log_dir = os.path.join(xp_log_dir, xpname)
    print(f'logging results in {xp_log_dir}')
    os.makedirs(xp_log_dir, exist_ok=True)
    with open(os.path.join(xp_log_dir, 'config.yaml'), 'w+') as f:
        OmegaConf.save(config=cfg, f=f)

    # Apply the initialization strategy
    if cfg.problem.type != 'inpainting_squared_mask':
        mask = None
    if cfg.init_strategy == 'y_noise':
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        save_image(x_init*0.5 + 0.5, os.path.join(xp_log_dir, 'x_init.png'))
        with torch.no_grad():
            qz = pipe.vae.encode(x_init.clip(-1, 1).half())
        mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor
        noise = torch.randn_like(mu_z)
        latents = pipe.scheduler.add_noise(mu_z, noise=noise, timesteps=torch.tensor([999]))

    elif cfg.init_strategy == 'y':
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        save_image(x_init*0.5 + 0.5, os.path.join(xp_log_dir, 'x_init.png'))
        with torch.no_grad():
            qz = pipe.vae.encode(x_init.clip(-1, 1).half())
        latents = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    text_embeddings_0 = text_embeddings

    accumulated_grad = torch.zeros_like(text_embeddings)

    csv_file = os.path.join(xp_log_dir, "metrics_log.csv")
    write_header = True  # Change to False if the file already exists and you don't want to rewrite the header.
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if it's the first time
        if write_header:
            writer.writerow(['Iteration', 'PSNR', 'SSIM', 'LPIPS'])

    start_time = time.time()
    for j in range(cfg.num_SAPG_steps):
        print(f"SAPG step: {j + 1}")
        # Inspecting the pipeline timesteps
        for i, timestep in enumerate(pipe.scheduler.timesteps):
            print(f"Step {i + 1}: Timestep {timestep}")
            text_embeddings = text_embeddings.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                noise_uncond = pipe.unet(
                    latents, 
                    timestep, 
                    encoder_hidden_states=text_embeddings, 
                    added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
                ).sample
            
            with torch.no_grad():
                _, noise_pred = noise_pred_cond_y_PRO(
                    latents=latents,
                    t = timestep,
                    pipe=pipe,
                    cfg=cfg,
                    logdir=xp_log_dir,
                    y_guidance=y_norm,
                    forward_model=forward_model,
                    noise_pred=noise_uncond,
                    sigma_y = sigma_y_norm,
                    SAPG_j = j,
                    n_steps = num_inference_steps
                )
            if i<num_inference_steps-1:
                alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                with torch.enable_grad():
                    z0_pred_c = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_uncond)

                with torch.no_grad():
                    latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

                def project_onto_ball(theta, theta_0, radius):
                    """Project theta onto a ball of radius `radius` centered at `theta_0`."""
                    delta = theta - theta_0
                    norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True)
                    scaling_factor = torch.clamp(radius / (norm_delta + 1e-8), max=1.0)
                    if (scaling_factor < 1.0).any():
                        print("Projected!")
                    return theta_0 + scaling_factor * delta
            
                with torch.enable_grad():
                    # Compute the loss
                    alpha_t = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[i+1]]
                    loss = -0.5/(1 - alpha_t) * torch.norm(latents - torch.sqrt(alpha_t)*z0_pred_c) ** 2

                    gradients = torch.autograd.grad(loss, inputs=text_embeddings, retain_graph=False)[0]
            
                accumulated_grad += gradients/torch.norm(gradients, dim=-1, keepdim=True)
                accumulated_grad[0,:4] = 0

                # Free memory
                accumulated_grad = accumulated_grad.detach()  # Detach from graph
                
            else:
                text_embeddings = project_onto_ball(text_embeddings + 0.08*(max(0.9**(max(0,j - 10)), 0.001)) * accumulated_grad, text_embeddings_0, 15)

                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                with torch.no_grad():
                    decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)
                    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)

                psnr = psnr_loss(restored_x, x_clean).item()
                ssim = ssim_loss(restored_x, x_clean).item()
                lpips = lpips_loss(restored_x * 2 - 1, x_clean * 2 -1).item()
                metrics = {
                    'PSNR' : psnr,
                    'SSIM' : ssim, 
                    'LPIPS': lpips
                }

                restored_x_lr = forward_model.A(restored_x.float())
                lr_psnr = psnr_loss(((y_norm+1)/2).clamp(0, 1), restored_x_lr).item()
                metrics['OBS-PSNR'] = lr_psnr

                metric_string = ""
                for m in metrics:
                    metric_string += f"{m}: {metrics[m]:.3f}, "
                print(metric_string)

                with open(csv_file, mode='a', newline='') as file:
                    # Write the metrics to the CSV file
                    writer = csv.writer(file)
                    writer.writerow([j + 1, psnr, ssim, lpips])

        if j<cfg.num_SAPG_steps-1:
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents2 = pipe.scheduler.add_noise(latents.detach(), noise=noise, timesteps=torch.tensor([999]))
            
                latents = latents2.clone()
                pipe.scheduler.set_timesteps(4, device=device)

                # Override the scheduler's timesteps with DMD2 values
                # custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
                custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
                pipe.scheduler.timesteps = custom_timesteps

                # Sample from the prior with the actual prompt c
                for _, timestep in enumerate(pipe.scheduler.timesteps):
                    noise_uncond = pipe.unet(
                        latents, 
                        timestep, 
                        encoder_hidden_states=text_embeddings, 
                        added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
                    ).sample
                    latents = pipe.scheduler.step(noise_uncond, timestep, latents).prev_sample

                # Decode latents to image
                decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

                save_image(torch.clamp(decoded_image * 0.5 + 0.5, 0, 1), os.path.join(xp_log_dir, f'prior_{j}.png'))
                # Prepare initial noise for next entry

                latents = latents2.clone()
                pipe.scheduler.set_timesteps(num_inference_steps, device=device)
                # Override the scheduler's timesteps with DMD2 values
                # custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
                custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
                pipe.scheduler.timesteps = custom_timesteps
        else:
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents2 = pipe.scheduler.add_noise(latents.detach(), noise=noise, timesteps=torch.tensor([999]))
            
                latents = latents2.clone()
                
                pipe.scheduler.set_timesteps(4, device=device)

                # Override the scheduler's timesteps with DMD2 values
                # custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
                custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
                pipe.scheduler.timesteps = custom_timesteps

                # Sample from the prior with the actual prompt c
                for k, timestep in enumerate(pipe.scheduler.timesteps):
                    noise_uncond = pipe.unet(
                        latents, 
                        timestep, 
                        encoder_hidden_states=text_embeddings, 
                        added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
                    ).sample
                    latents = pipe.scheduler.step(noise_uncond, timestep, latents).prev_sample

                with torch.no_grad():
                    # Decode latents to image
                    decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

                save_image(torch.clamp(decoded_image * 0.5 + 0.5, 0, 1), os.path.join(xp_log_dir, f'prior_{j}.png'))
                # Prepare initial noise for next entry

                latents = latents2.clone()
                del latents2
                
            pipe.scheduler.set_timesteps(8, device=device)
                
            # Override the scheduler's timesteps with DMD2 values
            custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
            #custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
            pipe.scheduler.timesteps = custom_timesteps

            for i, timestep in enumerate(pipe.scheduler.timesteps):
                with torch.no_grad():
                    noise_uncond = pipe.unet(
                        latents, 
                        timestep, 
                        encoder_hidden_states=text_embeddings, 
                        added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
                    ).sample
                
                with torch.no_grad():
                    _, noise_pred = noise_pred_cond_y_PRO(
                        latents=latents,
                        t = timestep,
                        pipe=pipe,
                        cfg=cfg,
                        logdir=xp_log_dir,
                        y_guidance=y_norm,
                        forward_model=forward_model,
                        noise_pred=noise_uncond,
                        sigma_y = sigma_y_norm,
                        SAPG_j = j,
                        n_steps = 8
                    )
                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                
    # End the timer
    end_time = time.time()

    # Print execution time
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

    # Load the metrics from the CSV file
    csv_file = os.path.join(xp_log_dir, "metrics_log.csv")
    metrics_data = pd.read_csv(csv_file)

    # Define the figure size for each individual plot
    fig_size = (6, 5)

    # PSNR Plot
    plt.figure(figsize=fig_size)
    plt.plot(metrics_data['Iteration'], metrics_data['PSNR'], label='PSNR', color='b', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.title('PSNR Over Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(xp_log_dir, "PSNR_plot.png"), dpi=300)
    plt.close()

    # SSIM Plot
    plt.figure(figsize=fig_size)
    plt.plot(metrics_data['Iteration'], metrics_data['SSIM'], label='SSIM', color='g', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.title('SSIM Over Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(xp_log_dir, "SSIM_plot.png"), dpi=300)
    plt.close()

    # LPIPS Plot
    plt.figure(figsize=fig_size)
    plt.plot(metrics_data['Iteration'], metrics_data['LPIPS'], label='LPIPS', color='r', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('LPIPS')
    plt.title('LPIPS Over Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(xp_log_dir, "LPIPS_plot.png"), dpi=300)
    plt.close()

    
    with torch.no_grad():
        # Decode latents to image
        decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)  # Normalize latents to image space

    save_image(restored_x, os.path.join(xp_log_dir, "restored.png"))
    save_image(((y_norm+1)/2).clamp(0, 1).detach().cpu(), os.path.join(xp_log_dir, "degraded.png"))
    save_image(x_clean.detach().cpu(), os.path.join(xp_log_dir, "clean.png"))

    psnr = psnr_loss(restored_x, x_clean).item()
    ssim = ssim_loss(restored_x, x_clean).item()
    lpips = lpips_loss(restored_x * 2 - 1, x_clean * 2 -1).item()
    metrics = {
        'PSNR' : psnr,
        'SSIM' : ssim, 
        'LPIPS': lpips
    }

    if type(forward_model) == dinv.physics.Downsampling:
        restored_x_lr = forward_model.A(restored_x.float())
        lr_psnr = psnr_loss(((y_norm+1)/2).clamp(0, 1), restored_x_lr).item()
        metrics['OBS-PSNR'] = lr_psnr

    metric_string = ""
    for m in metrics:
        metric_string += f"{m}: {metrics[m]:.3f}, "
    print(metric_string)

    with open(os.path.join(xp_log_dir, 'metrics.csv'), 'w+') as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    main()
