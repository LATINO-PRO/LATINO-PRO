import deepinv as dinv
import torch

from motionblur import Kernel


def disk_blur(radius):
    grid = torch.arange(2 * radius + 7) - (2 * radius + 7) // 2
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    disk = ((xx ** 2 + yy ** 2).float().sqrt() <= radius).float()
    return (disk / disk.sum()).unsqueeze(0).unsqueeze(0)


def get_forward_model(cfg, x_clean, device):
    noise_model = dinv.physics.GaussianNoise(sigma=cfg.problem.sigma_y)

    if cfg.problem.type == 'inpainting_squared_mask':
        B, C, H, W = x_clean.shape
        mask = torch.ones((1, H, W), device=x_clean.device)
        size = cfg.problem.mask_size
        # Define mask region
        mask[:, H // 2 - size // 5 - 35:H // 2 + size // 5 - 35,
        W // 2 - 4 * size // 5 - 2:W // 2 + 4 * size // 5 + 2] = 0
        forward_model = dinv.physics.Inpainting(
            tensor_size=x_clean.shape,
            mask=mask,
            noise_model=noise_model).to(device)
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'deblurring_gaussian':
        ksize = cfg.problem.sigma_kernel
        filter = dinv.physics.blur.gaussian_blur(sigma=(ksize, ksize))
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=filter,
            device=device,
            noise_model=noise_model)
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'deblurring_motion':
        kernel = Kernel(size=(122, 122), intensity=0.5)
        kernel_torch = torch.tensor(kernel.kernelMatrix, dtype=torch.float32)
        kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(0).to(device)
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=kernel_torch,
            device=device,
            noise_model=noise_model)
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'deblurring_disk':
        filter = disk_blur(cfg.problem.radius)
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=filter,
            device=device,
            noise_model=noise_model)
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'deblurring_aniso':
        filter = dinv.physics.blur.gaussian_blur(
            sigma=(cfg.problem.sigma_major, cfg.problem.sigma_minor),
            angle=cfg.problem.angle)
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=filter,
            device=device,
            noise_model=noise_model)
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'super_resolution_bicubic':
        forward_model = dinv.physics.Downsampling(
            img_size=x_clean.shape[1:],
            factor=cfg.problem.downscaling_factor,
            device=device,
            noise_model=noise_model,
            filter='bicubic',
            padding='reflect',
        )
        transpose_operator = forward_model.A_adjoint

    else:
        raise ValueError(f"Unexpected problem.type {cfg.problem.type}")

    return forward_model, transpose_operator
