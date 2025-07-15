# noise_schemes.py
import os
import torch
from torchvision.utils import save_image

def noise_pred_cond_y(
        latents,
        t: int,
        pipe,
        cfg,
        logdir,
        y_guidance,
        forward_model,
        noise_pred,
        sigma_y
    ):           
        with torch.no_grad(): 
            # Compute z0_pred
            alpha_t = pipe.scheduler.alphas_cumprod[t]
            z0_pred = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_pred)

            # decode
            x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)
            
        df = torch.norm(forward_model(x.float()) - y_guidance).item()
        # Extra regularization term
        var_x_zt = 1-alpha_t
        if cfg.problem.type == "super_resolution_bicubic":
            if cfg.problem.downscaling_factor == 16:
                if t>300:
                    delta = 6*0.01*df/(1e2*sigma_y)
                else:
                    delta = 9*0.01*df/(1e2*sigma_y)
            elif cfg.problem.downscaling_factor == 32:
                if t>300:
                    delta = 1.5*df/(1e1)
                else:
                    delta = 3*df/(1e1)
        elif cfg.problem.type == 'deblurring_gaussian':
            if cfg.problem.sigma_kernel<10:
                if t>400:
                    delta = 5*df/(1e4)
                else:
                    delta = 2*df/(1e4)
            else:
                if t>400:
                    delta = 7*df/(1e4)
                else:
                    delta = 3*df/(1e4)
        elif cfg.problem.type == 'deblurring_motion':
            if cfg.problem.sigma_y==0.01:
                if t>400:
                    delta = 4*df/(1e4)
                else:
                    delta = 2*df/(1e4)
            else:
                if t>400:
                    delta = 5*df/(1e5)
                elif t>0:
                    delta = 9*df/(1e5)
                else:
                    delta = 2*df/(1e7)
        elif cfg.problem.type == 'inpainting_squared_mask':
            if t>500:
                delta = 1
            else:
                delta = 0.5
        else:
            if t>200:
                delta = 0.01
            else:
                delta = 1
        print(f"delta at step {t}: ", "%.2f" % delta)
        with torch.no_grad():
            gamma = delta*var_x_zt/(sigma_y**2)
            gamma = gamma.to(device=latents.device)
            prox_x = forward_model.prox_l2(x.float(), y=y_guidance, gamma=gamma)

            # encode
            qz= pipe.vae.encode(prox_x.clip(-1,1).half())
            mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor

            z0_pred_cond_y = mu_z

            noise_pred_cond_y = torch.sqrt(1/(1-alpha_t))*latents - torch.sqrt(alpha_t/(1-alpha_t))*z0_pred_cond_y
        log_image_dict = {'x': x, 'prox': prox_x}

        logdir_iter = os.path.join(logdir, 'iter')
        os.makedirs(logdir_iter, exist_ok=True)
        
        for k, v in log_image_dict.items():
            save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
            
        return z0_pred_cond_y, noise_pred_cond_y

def noise_pred_cond_y_15(
    latents,
    t: int,
    encoder_hidden_states,
    guidance_scale,
    pipe,
    cfg,
    logdir,
    y_guidance,
    forward_model,
    sigma_y
):
    with torch.no_grad():
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # Format timestep correctly
        t_tensor = torch.tensor([t], dtype=torch.float16).to("cuda")

        # Forward pass through UNet
        noise_pred = pipe.unet(
            latent_model_input, 
            t_tensor, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Split the outputs for CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # Compute z0_pred
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_pred)

    # decode
    with torch.no_grad():
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)
    df = torch.norm(forward_model(x.float()) - y_guidance).item()
    var_x_zt = 1-alpha_t
    if cfg.problem.type == "super_resolution_bicubic":
        if cfg.problem.downscaling_factor == 16:
            if t>300:
                delta = 1*0.02*df/(1e0*sigma_y)
            else:
                delta = 1*0.02*df/(1e1*sigma_y)
        elif cfg.problem.downscaling_factor == 32:
            if t>300:
                delta = 2*df/(1e0)
            else:
                delta = 9*df/(1e1)
    elif cfg.problem.type == 'inpainting_squared_mask':
        if t>500:
            delta = 1
        else:
            delta = 0.5
    elif cfg.problem.type == 'deblurring_gaussian':
        if t>300:
            delta = 1*df/(1e3)
        else:
            delta = 4*df/(1e4)
    elif cfg.problem.type == 'deblurring_motion':
        if t>400:
            delta = 8*df/(1e4)
        else:
            delta = 7*df/(1e4)
    else:
        if t>200:
            delta = 0.01
        else:
            delta = 1
    print(f"delta at step {t}: ", "%.2f" % delta)
    with torch.no_grad():
        prox_x = forward_model.prox_l2(x.float().detach().clone(), y=y_guidance, gamma=delta*var_x_zt/(sigma_y**2))
    # encode
    with torch.no_grad():
        qz= pipe.vae.encode(prox_x.clip(-1,1).half())
    mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    z0_pred_cond_y = mu_z

    noise_pred_cond_y = torch.sqrt(1/(1-alpha_t))*latents - torch.sqrt(alpha_t/(1-alpha_t))*z0_pred_cond_y
    log_image_dict = {'x': x, 'prox': prox_x}

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    
    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
        
    return noise_pred_cond_y

def noise_pred_cond_y_DPS(
    latents,
    t: int,
    encoder_hidden_states,
    guidance_scale,
    pipe,
    logdir,
    y_guidance,
    forward_model
):
    with torch.enable_grad():
        latents = latents.detach().requires_grad_(True)

        # Expand latents for unconditional/conditional input for CFG
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # Format timestep correctly
        t_tensor = torch.tensor([t], dtype=torch.float16).to("cuda")

        # Forward pass through UNet
        noise_pred = pipe.unet(
            latent_model_input, 
            t_tensor, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Split the outputs for CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1/alpha_t)*(latents - torch.sqrt(1-alpha_t)*noise_pred)

        # compute approximate log likelihood ||AD(z_0) -latents ||^2 / (2 sigma**2)
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor).sample

        nlogpyx = torch.linalg.norm((forward_model.A(x.float())-y_guidance))
        print("loss: ", nlogpyx.item())
        # compute neg log liklihood gradient
    grad_nll = torch.autograd.grad(nlogpyx, latents)[0]

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    log_image_dict = {'x': x}

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
    return noise_pred, grad_nll

def noise_pred_cond_y_PSLD(
    latents,
    t: int,
    encoder_hidden_states,
    guidance_scale,
    pipe,
    logdir,
    y_guidance,
    forward_model,
    transpose_model
):
    with torch.enable_grad():
        latents = latents.detach().requires_grad_(True)

        # Expand latents for unconditional/conditional input for CFG
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # Format timestep correctly
        t_tensor = torch.tensor([t], dtype=torch.float16).to("cuda")

        # Forward pass through UNet
        noise_pred = pipe.unet(
            latent_model_input, 
            t_tensor, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Split the outputs for CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1/alpha_t)*(latents - torch.sqrt(1-alpha_t)*noise_pred)

        # compute approximate log likelihood ||AD(z_0) -latents ||^2 / (2 sigma**2)
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor).sample

        meas_pred = forward_model.A(x.float())
        meas_error = torch.linalg.norm((meas_pred-y_guidance))
        print("loss: ", meas_error.item())
        # This computes x_0*
        ortho_project = x.float() - transpose_model(meas_pred)
        parallel_project = transpose_model(y_guidance)
        inpainted_image = parallel_project + ortho_project
        
        encoded_z_0 = pipe.vae.encode(inpainted_image.type(torch.float16).clip(-1,1)).latent_dist.mean * pipe.vae.config.scaling_factor
        inpaint_error = torch.linalg.norm((encoded_z_0 - z0_pred))
        print("gluing loss: ", inpaint_error.item())

        gamma, omega = 1e-1, 1
        error = inpaint_error * gamma + meas_error * omega
    gradients = torch.autograd.grad(error, inputs = latents)[0]

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    log_image_dict = {'x': x}

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
    return noise_pred, gradients

def noise_pred_cond_y_DPS_P2L(
    latents,
    t: int,
    encoder_hidden_states,
    guidance_scale,
    pipe,
    logdir,
    y_guidance,
    forward_model
):
    with torch.enable_grad():
        latents = latents.detach().requires_grad_(True)

        # Expand latents for unconditional/conditional input for CFG
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # Format timestep correctly
        t_tensor = torch.tensor([t], dtype=torch.float16).to("cuda")

        # Forward pass through UNet
        noise_pred = pipe.unet(
            latent_model_input, 
            t_tensor, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Split the outputs for CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1/alpha_t)*(latents - torch.sqrt(1-alpha_t)*noise_pred)

        # compute approximate log likelihood ||AD(z_0) -latents ||^2 / (2 sigma**2)
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor).sample.clip(-1, 1)

        nlogpyx = torch.linalg.norm((forward_model.A(x.float())-y_guidance))
        print("loss: ", nlogpyx.item())
        # compute neg log liklihood gradient
    grad_nll = torch.autograd.grad(nlogpyx, latents)[0]

    # modify hyperparm according to Table 6 in the paper: # https://arxiv.org/pdf/2310.01110
    if t%8 == 1:
        with torch.no_grad(): 
            prox_x = forward_model.prox_l2(x.float(), y=y_guidance, gamma=1)

            # encode
            qz= pipe.vae.encode(prox_x.clip(-1,1).half())
            z0_pred = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    noise_pred_cond_y = torch.sqrt(1/(1-alpha_t))*latents - torch.sqrt(alpha_t/(1-alpha_t))*z0_pred.detach()

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    log_image_dict = {'x': x}

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
    return noise_pred_cond_y, grad_nll

def noise_pred_cond_y_DPS_1024(
    latents,
    t: int,
    text_embeddings,
    added_cond_kwargs,
    pipe,
    logdir,
    y_guidance,
    forward_model
):
    with torch.enable_grad():
        latents = latents.detach().requires_grad_(True)
        noise_pred = pipe.unet(
            latents, 
            t, 
            encoder_hidden_states=text_embeddings, 
            added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
        ).sample
        
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1/alpha_t)*(latents - torch.sqrt(1-alpha_t)*noise_pred)

        # compute approximate log likelihood ||AD(z_0) -latents ||^2 / (2 sigma**2)
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor).sample

        nlogpyx = torch.linalg.norm((forward_model.A(x.float())-y_guidance))
        print("loss: ", nlogpyx.item())
        # compute neg log liklihood gradient
    grad_nll = torch.autograd.grad(nlogpyx, latents)[0]

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    log_image_dict = {'x': x}

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
    return noise_pred, grad_nll

def noise_pred_cond_y_PSLD_1024(
    latents,
    t: int,
    text_embeddings,
    added_cond_kwargs,
    pipe,
    logdir,
    y_guidance,
    forward_model,
    transpose_model,
):
    with torch.enable_grad():
        latents = latents.detach().requires_grad_(True)
        noise_pred = pipe.unet(
            latents, 
            t, 
            encoder_hidden_states=text_embeddings, 
            added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
        ).sample
        
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1/alpha_t)*(latents - torch.sqrt(1-alpha_t)*noise_pred)

        # compute approximate log likelihood ||AD(z_0) -latents ||^2 / (2 sigma**2)
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor).sample

        meas_pred = forward_model.A(x.float())
        meas_error = torch.linalg.norm((meas_pred-y_guidance))
        print("loss: ", meas_error.item())
        # This computes x_0*
        ortho_project = x.float() - transpose_model(meas_pred)
        parallel_project = transpose_model(y_guidance)
        inpainted_image = parallel_project + ortho_project
        
        encoded_z_0 = pipe.vae.encode(inpainted_image.clip(-1,1)).latent_dist.mean * pipe.vae.config.scaling_factor
        inpaint_error = torch.linalg.norm((encoded_z_0 - z0_pred))
        print("gluing loss: ", inpaint_error.item())

        gamma, omega = 1e-1, 1
        error = inpaint_error * gamma + meas_error * omega
    gradients = torch.autograd.grad(error, inputs = latents)[0]

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    log_image_dict = {'x': x}

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
    return noise_pred, gradients

def noise_pred_cond_y_DPS_1024_P2L(
    latents,
    t: int,
    text_embeddings,
    added_cond_kwargs,
    pipe,
    logdir,
    y_guidance,
    forward_model
):
    with torch.enable_grad():
        latents = latents.detach().requires_grad_(True)
        noise_pred = pipe.unet(
            latents, 
            t, 
            encoder_hidden_states=text_embeddings, 
            added_cond_kwargs=added_cond_kwargs  # Include additional conditioning
        ).sample
        
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1/alpha_t)*(latents - torch.sqrt(1-alpha_t)*noise_pred)

        # compute approximate log likelihood ||AD(z_0) -latents ||^2 / (2 sigma**2)
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor).sample

        nlogpyx = torch.linalg.norm((forward_model.A(x.float())-y_guidance))
        print("loss: ", nlogpyx.item())
        # compute neg log liklihood gradient
    grad_nll = torch.autograd.grad(nlogpyx, latents)[0]

    # modify hyperparm according to Table 6 in the paper: # https://arxiv.org/pdf/2310.01110
    if t%8 == 1:
        with torch.no_grad(): 
            prox_x = forward_model.prox_l2(x.float(), y=y_guidance, gamma=1)

            # encode
            qz= pipe.vae.encode(prox_x.clip(-1,1))
            z0_pred = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    noise_pred_cond_y = torch.sqrt(1/(1-alpha_t))*latents - torch.sqrt(alpha_t/(1-alpha_t))*z0_pred.detach()

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    log_image_dict = {'x': x}

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}.png'))
    return noise_pred_cond_y, grad_nll

def noise_pred_cond_y_TReg(
    x,
    z0_pred,
    pipe,
    y_guidance,
    forward_model,
):
    with torch.no_grad():
        with torch.no_grad():
            prox_x = forward_model.prox_l2(x.float().detach().clone(), y=y_guidance, gamma=1e4) # gamma=1/lambda used in TReg
        # encode
        with torch.no_grad():
            qz= pipe.vae.encode(prox_x.clip(-1,1).half())
        z0_pred = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    return z0_pred, prox_x

def noise_pred_cond_y_PRO(
    latents,
    t: int,
    pipe,
    cfg,
    logdir,
    y_guidance,
    forward_model,
    noise_pred,
    sigma_y,
    SAPG_j,
    n_steps = 4
):           
    with torch.no_grad(): 
        # Compute z0_pred
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_pred)

        # decode
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)
        
    df = torch.norm(forward_model(x.float()) - y_guidance).item()
    var_x_zt = 1-alpha_t
    if cfg.problem.type == "super_resolution_bicubic":
        if cfg.problem.sigma_y < 0.05:
            if cfg.problem.downscaling_factor == 16:
                if t>300:
                    delta = 6*0.01*df/(1e2*sigma_y)
                else:
                    delta = 9*0.01*df/(1e2*sigma_y)
            elif cfg.problem.downscaling_factor == 32:
                if t>300:
                    delta = 1.5*0.01*df/(1e0*sigma_y)
                else:
                    delta = 9*0.01*df/(1e1*sigma_y)
            else:
                delta = 1
        else:
            if cfg.problem.downscaling_factor == 16:
                if n_steps ==4:
                    if t>500:
                        delta = 5*0.01*df/(1e0*sigma_y)
                    else:
                        delta = 8*0.01*df/(1e1*sigma_y)
                else:
                    if t>700:
                        delta = 6*0.01*df/(1e0*sigma_y)
                    else:
                        delta = 2*0.01*df/(1e0*sigma_y)
            elif cfg.problem.downscaling_factor == 32:
                if t>300:
                    delta = 15*0.01*df/(1e0*sigma_y)
                else:
                    delta = 9*0.01*df/(1e0*sigma_y)
            else:
                delta = 1
    elif cfg.problem.type == 'deblurring_gaussian':
        if cfg.problem.sigma_y < 0.05 and cfg.problem.sigma_kernel < 10:
            if n_steps ==4:
                if t>500:
                    delta = 4*df/(1e3)
                else:
                    delta = 2*df/(1e3)
            else:
                if t>400:
                    delta = 4*df/(1e4)
                else:
                    delta = 4*df/(1e4)
            
        elif cfg.problem.sigma_y < 0.05 and cfg.problem.sigma_kernel > 10:
            if n_steps ==4:
                if t>400:
                    delta = 2*df/(1e2)
                else:
                    delta = 2*df/(1e3)
            else:
                if t>400:
                    delta = 2*df/(1e2)
                else:
                    delta = 2*df/(1e3)
        else:
            if n_steps ==4:
                if t>500:
                    delta = 8*df/(1e3)
                else:
                    delta = 5*df/(1e3)
            else:
                if t>400:
                    delta = 8*df/(1e3)
                else:
                    delta = 5*df/(1e3)
    elif cfg.problem.type == 'deblurring_motion':
        if n_steps ==4:
            if t>500:
                delta = 4*df/(1e4)
            else:
                delta = 5*df/(1e4)
        else:
            if t>400:
                delta = 5*df/(1e4)
            else:
                delta = 9*df/(1e4)
    else:
        if t>200:
            delta = 0.01
        else:
            delta = 1
    print(f"delta at step {t}: ", "%.2f" % delta)

    with torch.no_grad():
        prox_x = forward_model.prox_l2(x.float(), y=y_guidance, gamma=delta*var_x_zt/(sigma_y**2))

        # encode
        qz= pipe.vae.encode(prox_x.clip(-1,1).half())
        mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    z0_pred_cond_y = mu_z

    noise_pred_cond_y = torch.sqrt(1/(1-alpha_t))*latents - torch.sqrt(alpha_t/(1-alpha_t))*z0_pred_cond_y
    log_image_dict = {'x': x, 'prox': prox_x}

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)
    
    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), os.path.join(logdir_iter, f'{t:3d}_{k}_{SAPG_j}.png'))
        
    return z0_pred_cond_y, noise_pred_cond_y
