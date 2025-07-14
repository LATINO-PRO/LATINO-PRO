import os
import json
import time
import deepinv as dinv
from torchvision.utils import save_image
import random
import torch
import torchmetrics
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
    LCMScheduler,
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusionPipeline,
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
    noise_pred_cond_y,
    noise_pred_cond_y_15,
    noise_pred_cond_y_DPS,
    noise_pred_cond_y_PSLD,
    noise_pred_cond_y_DPS_P2L,
    noise_pred_cond_y_DPS_1024,
    noise_pred_cond_y_PSLD_1024,
    noise_pred_cond_y_DPS_1024_P2L,
    noise_pred_cond_y_TReg,
)

@hydra.main(version_base=None, config_path="configs", config_name="LATINO")
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

    if cfg.model == "TREG":
        # Load CLIP ViT-L/14
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # load stable diffusion
    if cfg.model in {"LATINO", "LDPS1024", "PSLD1024", "LDPS1024-P2L"}:
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        if cfg.model == "LATINO":
            repo_name = "tianweiy/DMD2"
            ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
            unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
            unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
            unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device, weights_only=True))
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16", guidance_scale=0).to(device)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        else:
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

            # ───────────────────────────────────────────────────────────
            #  Build the pipeline
            # ───────────────────────────────────────────────────────────
            pipe = DiffusionPipeline.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,  #full precision to avoid float16 issues with autograd
                guidance_scale=0
            ).to(device)

            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        prompt = cfg.image.prompt

        # Encode text to conditioning
        text_embeddings, _, pooled_text_embeds, _ = pipe.encode_prompt(
            prompt,
            device=device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False
        )

        uncond_embeddings, _, _, _ = pipe.encode_prompt(
            "",
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
            dtype=torch.float16 if cfg.model == "LATINO" else torch.float32,  # datatype
            device=device,  # Corrected device type
            generator=generator  # Random number generator
        )

        # Get time_ids automatically based on the image resolution
        time_ids = pipe._get_add_time_ids(
            original_size=(image_height, image_width),  # The original image resolution
            crops_coords_top_left=(0, 0),  # No cropping
            target_size=(image_height, image_width),  # Target resolution
            dtype=torch.float16 if cfg.model == "LATINO" else torch.float32,  # Ensure correct data type
            text_encoder_projection_dim=1280
        ).to(device)
        
        # Additional conditioning required for SDXL
        added_cond_kwargs = {
            "text_embeds": pooled_text_embeds,  # Pass the pooled text embeddings
            "time_ids": time_ids
        }

        # Define the number of inference steps and set timesteps
        if cfg.model == "LATINO":
            num_inference_steps = 8
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)

            custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
            #custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
            pipe.scheduler.timesteps = custom_timesteps
        else:
            num_inference_steps = 500
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    else:
        # Load Stable Diffusion v1.5 components
        if cfg.model != "LATINO-1.5":
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            if cfg.model == "TREG":
                num_inference_steps = 200
                guidance_scale = 5.0 # CFG scale
            else:
                num_inference_steps = 999
                guidance_scale = 1  # CFG scale
        else:
            model_id = "runwayml/stable-diffusion-v1-5"
            adapter_id = "latent-consistency/lcm-lora-sdv1-5"

            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")

            # load and fuse lcm lora
            pipe.load_lora_weights(adapter_id)
            pipe.fuse_lora()

            num_inference_steps = 8
            guidance_scale = 1  # CFG scale

        # Extract individual components
        unet = pipe.unet
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        # Define the prompt
        prompt = [cfg.image.prompt]

        # Encode the prompt to conditioning embeddings
        text_inputs = tokenizer(
            prompt, 
            padding="max_length", 
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

        # Create unconditional (empty) prompt embeddings for CFG
        if cfg.model == "TREG":
            uncond_inputs = tokenizer(
                ["out of focus, depth of field"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
        else:
            uncond_inputs = tokenizer(
                [""] * len(prompt),  # Empty prompt for unconditional guidance
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

        # Encode unconditional prompt
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]

        # Set diffusion parameters
        height, width = 512, 512  # Image resolution

        # Prepare the latent space (initial random noise)
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, height // 8, width // 8),
            device="cuda",
            dtype=torch.float16,
        )

        # Initialize scheduler and set the number of inference steps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        # Concatenate unconditional and conditional embeddings for CFG
        text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    # load clean image
    xtemp = load_image_tensor(cfg.image.path)

    x_clean = crop_to_multiple(xtemp, m=8).to(device)
    
    # To adapt the method to 1024x1024 images in case of SDv1.5
    if cfg.model not in {"LATINO", "LDPS1024", "PSLD1024", "LDPS1024-P2L"}:
        if xtemp.shape[-1] == 1024:
            noise_model_1024_to_512 = dinv.physics.GaussianNoise(sigma=0)

            model_1024_to_512 = dinv.physics.Downsampling(
                img_size=(3, 1024, 1024),
                factor=2,
                device=device,
                noise_model=noise_model_1024_to_512,
                filter = "bicubic"
                )
        
            x_clean = model_1024_to_512(x_clean).clamp(0,1)
    # To adapt the method to 512x512 images in case of DMD2
    else:
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

    if cfg.model == "LATINO":
        xp_log_dir = os.path.join(logdir, "results_LATINO", cfg.problem.type, cfg.log_subfolder)
    elif cfg.model in {"PSLD", "PSLD1024"}:
        xp_log_dir = os.path.join(logdir, "results_PSLD", cfg.problem.type, cfg.log_subfolder)
    elif cfg.model == "LATINO-1.5":
        xp_log_dir = os.path.join(logdir, "results_LATINO_1.5", cfg.problem.type, cfg.log_subfolder)
    elif cfg.model in {"LDPS", "LDPS1024"}:
        xp_log_dir = os.path.join(logdir, "results_LDPS", cfg.problem.type, cfg.log_subfolder)
    elif cfg.model in {"LDPS-P2L", "LDPS1024-P2L"}:
        xp_log_dir = os.path.join(logdir, "results_P2L", cfg.problem.type, cfg.log_subfolder)
    elif cfg.model in {"LDPS-P2L", "LDPS1024-P2L"}:
        xp_log_dir = os.path.join(logdir, "results_LDPS_P2L", cfg.problem.type, cfg.log_subfolder)
    elif cfg.model == "TREG":
        xp_log_dir = os.path.join(logdir, "results_TREG", cfg.problem.type, cfg.log_subfolder)

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
            qz = pipe.vae.encode(x_init.clip(-1, 1).half()) if cfg.model not in {"LDPS1024-P2L", "LDPS1024", "PSLD1024"} else pipe.vae.encode(x_init.clip(-1, 1))
        mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor
        noise = torch.randn_like(mu_z)
        latents = pipe.scheduler.add_noise(mu_z, noise=noise, timesteps=torch.tensor([999]))

    elif cfg.init_strategy == 'y':
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        save_image(x_init*0.5 + 0.5, os.path.join(xp_log_dir, 'x_init.png'))
        with torch.no_grad():
            qz = pipe.vae.encode(x_init.clip(-1, 1).half()) if cfg.model not in {"LDPS1024-P2L", "LDPS1024", "PSLD1024"} else pipe.vae.encode(x_init.clip(-1, 1))
        latents = qz.latent_dist.mean * pipe.vae.config.scaling_factor
    
    start_time = time.time()

    if cfg.model in {"LDPS-P2L", "LDPS1024-P2L"}:
        # Adam parameters for P2L
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 1e-4  # Learning rate for prompt
        lr2 = 0.05   # Learning rate for gradient

        # Initialize moment estimates
        m = torch.zeros_like(text_embeddings, dtype=torch.float32)
        v = torch.zeros_like(text_embeddings, dtype=torch.float32)
        t_step = 0  # Time step for bias correction

        # Initialize moment estimates
        m2 = torch.zeros_like(latents, dtype=torch.float32)
        v2 = torch.zeros_like(latents, dtype=torch.float32)
        t_step2 = 0  # Time step for bias correction

    # Inspecting the pipeline timesteps
    for i, timestep in enumerate(pipe.scheduler.timesteps):
        print(f"Step {i + 1}: Timestep {timestep}")
        if cfg.model == "LATINO":
            text_embeddings = text_embeddings.detach().requires_grad_(True)
            with torch.no_grad():
                noise_uncond = pipe.unet(
                    latents, 
                    timestep, 
                    encoder_hidden_states=text_embeddings, 
                    added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
                ).sample

            with torch.no_grad():
                _, noise_pred = noise_pred_cond_y(
                    latents=latents,
                    t = timestep,
                    pipe=pipe,
                    cfg=cfg,
                    logdir=xp_log_dir,
                    y_guidance=y_norm,
                    forward_model=forward_model,
                    noise_pred=noise_uncond,
                    sigma_y=sigma_y_norm
                )
        elif cfg.model == "LATINO-1.5":
            with torch.no_grad():
                noise_pred = noise_pred_cond_y_15(
                    latents=latents,
                    t=timestep,
                    encoder_hidden_states=text_embeddings_cfg,
                    guidance_scale=guidance_scale,
                    pipe=pipe,
                    cfg=cfg,
                    logdir=xp_log_dir,
                    y_guidance=y_norm,
                    forward_model=forward_model,
                    sigma_y=sigma_y_norm
                )
        elif cfg.model == "LDPS":
            noise_pred, grad_nll = noise_pred_cond_y_DPS(
                latents=latents,
                t=timestep,
                encoder_hidden_states=text_embeddings_cfg,
                guidance_scale=guidance_scale,
                pipe=pipe,
                logdir=xp_log_dir,
                y_guidance=y_norm,
                forward_model=forward_model
            )
        elif cfg.model == "LDPS1024":
            noise_pred, grad_nll = noise_pred_cond_y_DPS_1024(
                latents=latents,
                t = timestep,
                pipe=pipe,
                text_embeddings=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                logdir=xp_log_dir,
                y_guidance=y_norm,
                forward_model=forward_model
            )
        elif cfg.model == "PSLD":
            noise_pred, grad_nll = noise_pred_cond_y_PSLD(
                latents=latents,
                t=timestep,
                encoder_hidden_states=text_embeddings_cfg,
                guidance_scale=guidance_scale,
                pipe=pipe,
                logdir=xp_log_dir,
                y_guidance=y_norm,
                forward_model=forward_model,
                transpose_model=transpose_operator
            )
        elif cfg.model == "PSLD1024":
            # Call the denoiser with transformed latents           
            noise_pred, grad_nll = noise_pred_cond_y_PSLD_1024(
                latents=latents,
                t=timestep,
                pipe=pipe,
                text_embeddings=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                logdir=xp_log_dir,
                y_guidance=y_norm,
                forward_model=forward_model,
                transpose_model=transpose_operator
            )
        elif cfg.model == "LDPS-P2L":
            # Modify hyperparameters for P2L according to Table 6 in the paper: # https://arxiv.org/pdf/2310.01110
            for _ in range(1):
                text_embeddings = text_embeddings.detach().requires_grad_(True)
                text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings], dim=0)
                with torch.enable_grad():
                    # Expand latents for unconditional/conditional input for CFG
                    latent_model_input = torch.cat([latents] * 2, dim=0)

                    # Format timestep correctly
                    t_tensor = torch.tensor([timestep], dtype=torch.float16).to("cuda")

                    # Forward pass through UNet
                    noise_pred = pipe.unet(
                        latent_model_input, 
                        t_tensor, 
                        encoder_hidden_states=text_embeddings_cfg
                    ).sample

                    # Split the outputs for CFG
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_uncond = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    alpha_t = pipe.scheduler.alphas_cumprod[timestep]

                    z0_pred_c = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_uncond)
                    # decode
                    x = pipe.vae.decode(z0_pred_c / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)

                    # Rescale from [-1, 1] to [0, 1]
                    x = (x + 1) / 2

                    model_output = forward_model(x.float()).clamp(-1, 1)

                    loss = torch.norm(y_norm - model_output)

                    gradients = torch.autograd.grad(loss, inputs=text_embeddings)[0]
                
                # Adam update step (done in float32)
                t_step += 1
                m = beta1 * m + (1 - beta1) * gradients  # First moment estimate (momentum)
                v = beta2 * v + (1 - beta2) * (gradients ** 2)  # Second moment estimate (adaptive scaling)

                # Bias correction
                m_hat = m / (1 - beta1 ** t_step)
                v_hat = v / (1 - beta2 ** t_step)

                # **Fix: Ensure v_hat is never zero (prevents division by zero)**
                v_hat = torch.clamp(v_hat, min=epsilon)

                # Update text embeddings with Adam (convert back to float16)
                text_embeddings = text_embeddings - (lr / torch.sqrt(v_hat)) * m_hat
                text_embeddings = text_embeddings.to(torch.float16)
            
            noise_pred, grad_nll = noise_pred_cond_y_DPS_P2L(
                latents=latents,
                t=timestep,
                encoder_hidden_states=text_embeddings_cfg,
                guidance_scale=guidance_scale,
                pipe=pipe,
                logdir=xp_log_dir,
                y_guidance=y_norm,
                forward_model=forward_model
            )
        elif cfg.model == "LDPS1024-P2L":
            with torch.autocast(device_type="cuda", enabled=False):
                # Modify hyperparameters for P2L according to Table 6 in the paper: # https://arxiv.org/pdf/2310.01110
                for _ in range(5):
                    text_embeddings = text_embeddings.detach().requires_grad_(True)
                    text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings], dim=0)
                    with torch.enable_grad():
                        noise_uncond = pipe.unet(
                            latents, 
                            timestep, 
                            encoder_hidden_states=text_embeddings.float(), 
                            added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
                        ).sample

                        alpha_t = pipe.scheduler.alphas_cumprod[timestep]

                        z0_pred_c = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_uncond)
                        # decode
                        x = pipe.vae.decode(z0_pred_c / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)

                        # Rescale from [-1, 1] to [0, 1]
                        x = (x + 1) / 2

                        model_output = forward_model(x.float()).clamp(-1, 1)

                        loss = torch.norm(y_norm - model_output)

                        gradients = torch.autograd.grad(loss, inputs=text_embeddings, retain_graph=False)[0]

                    # Adam update step (done in float32)
                    t_step += 1
                    m = beta1 * m + (1 - beta1) * gradients  # First moment estimate (momentum)
                    v = beta2 * v + (1 - beta2) * (gradients ** 2)  # Second moment estimate (adaptive scaling)

                    # Bias correction
                    m_hat = m / (1 - beta1 ** t_step)
                    v_hat = v / (1 - beta2 ** t_step)

                    # **Fix: Ensure v_hat is never zero (prevents division by zero)**
                    v_hat = torch.clamp(v_hat, min=epsilon)

                    # Update text embeddings with Adam (convert back to float16)
                    text_embeddings = text_embeddings - (lr / torch.sqrt(v_hat)) * m_hat

            noise_pred, grad_nll = noise_pred_cond_y_DPS_1024_P2L(
                latents=latents,
                t = timestep,
                pipe=pipe,
                text_embeddings=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                logdir=xp_log_dir,
                y_guidance=y_norm,
                forward_model=forward_model
            )
        elif cfg.model == "TREG":
            # This is a deepinverse-compatible implementation of TReg. See https://arxiv.org/pdf/2311.15658 for more details
            with torch.no_grad():
                skip = 5
                prev_t = timestep - skip
                alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                at_prev = pipe.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else pipe.scheduler.final_alpha_cumprod.to(device)
                # Expand latents for unconditional/conditional input for CFG
                latent_model_input = torch.cat([latents] * 2, dim=0)

                # Format timestep correctly
                t_tensor = torch.tensor(timestep, dtype=torch.float16).to("cuda")

                t_in = torch.cat([t_tensor.unsqueeze(0)] * 2)
                
                text_embeddings_cfg = torch.cat([uncond_embeddings.half(), text_embeddings], dim=0)
                # Forward pass through UNet
                noise_pred = pipe.unet(
                    latent_model_input, 
                    t_in, 
                    encoder_hidden_states=text_embeddings_cfg
                ).sample

                # Split the outputs for CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if timestep%3 == 0 and timestep<850:
                with torch.enable_grad():
                    # For TReg testing
                    with torch.no_grad():
                        #z0_pred_c = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_pred)

                        z0_pred_c = (latents - (1-alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                        
                        # decode
                        x = pipe.vae.decode(z0_pred_c / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)

                        z0_predy, x = noise_pred_cond_y_TReg(
                            x=x,
                            z0_pred=z0_pred_c,
                            pipe=pipe,
                            y_guidance=y_norm,
                            forward_model=forward_model
                        )

                        # Rescale from [-1, 1] to [0, 1]
                        x_clip = (x + 1) / 2
                        x_clip = x_clip.clip(0, 1)

                    # Apply preprocessing
                    image_tensor = clip_processor(images=x_clip.float(), return_tensors="pt").to(device)['pixel_values']

                    # Get features

                    img_feats = clip_model.get_image_features(pixel_values=image_tensor)
                    img_feats = img_feats / img_feats.norm(dim=1, keepdim=True) # normalize

                    # Output shapes
                    #print(f"OpenCLIP features shape: {openclip_features.shape}")
                    #print(f"CLIP ViT-L features shape: {clip_features.shape}")
                    #print(f"Combined features shape: {combined_features.shape}")
                    
                    lr = 1e-3  # Learning rate for prompt
                    uncond_embeddings = uncond_embeddings.to(torch.float32)
                    uncond_embeddings = uncond_embeddings.clone().detach().requires_grad_(True)
                    optim_text = torch.optim.Adam([uncond_embeddings], lr=lr)

                    for _ in range(10):
                        optim_text.zero_grad()

                        sim = img_feats @ uncond_embeddings.permute(0, 2, 1)
                        loss = sim.mean()

                        loss.backward(retain_graph=True)
                        optim_text.step()

                    noise = torch.randn_like(z0_predy).to(device)
                    z0_ema = at_prev * z0_predy + (1-at_prev) * z0_pred_c
                    latents = at_prev.sqrt() * z0_ema + (1-at_prev) * noise_pred
                    latents = latents + (1-at_prev).sqrt() * at_prev.sqrt() * noise
            else:
                with torch.no_grad():
                    z0t = (latents - (1-alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                    latents = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
                    x = pipe.vae.decode(z0t / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)
            # Log images
            logdir_iter = os.path.join(xp_log_dir, 'iter')
            os.makedirs(logdir_iter, exist_ok=True)
            log_image_dict = {'x': x}

            for k, v in log_image_dict.items():
                save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{timestep:3d}_{k}.png'))

        # step the scheduler
        if cfg.model != "TREG":
            latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

        if cfg.model in {"LDPS", "PSLD", "LDPS1024", "PSLD1024"}:
            latents -= grad_nll
        elif cfg.model in {"LDPS1024-P2L", "LDPS-P2L"}:
            if cfg.problem.type == "super_resolution_bicubic":
                latents -= lr2*grad_nll
            else:
                # Adam update
                t_step2 += 1
                m2 = beta1 * m2 + (1 - beta1) * grad_nll  # First moment estimate
                v2 = beta2 * v2 + (1 - beta2) * (grad_nll ** 2)  # Second moment estimate

                # Bias correction
                m_hat2 = m2 / (1 - beta1 ** t_step)
                v_hat2 = v2 / (1 - beta2 ** t_step)

                v_hat2 = torch.clamp(v_hat2, min=epsilon)

                # Update parameters
                latents = latents - (lr2 / (torch.sqrt(v_hat2) + epsilon)) * m_hat2
                if cfg.model in {"LDPS-P2L"}:
                    latents = latents.to(torch.float16)  # Convert back to float16

    
    with torch.no_grad():
        # Decode latents to image
        decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)  # Normalize latents to image space

    # End the timer
    end_time = time.time()

    # Print execution time
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

    save_image(restored_x, os.path.join(xp_log_dir, "restored.png"))
    save_image(((y_norm+1)/2).clamp(0, 1).detach().cpu(), os.path.join(xp_log_dir, "degraded.png"))
    save_image(x_clean.detach().cpu(), os.path.join(xp_log_dir, "clean.png"))

    lpips_loss = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device)
    
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
