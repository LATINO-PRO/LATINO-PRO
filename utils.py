import os
import torch
import numpy as np
from PIL import Image
import deepinv as dinv
import torch.nn.functional as F


def pt2np(x):
    return x.squeeze(0).permute(1, 2, 0).detach().cpu()

def pil2pt(img):
    return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0) / 255

def load_image_tensor(path): # TODO: torchvision?
    img = Image.open(path).convert('RGB')
    return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0) / 255

def load_image_tensor_imagenet(path, resize_to=(512, 512)):
    """Load an image from the given path, crop it to a centered square, resize, and return it as a normalized tensor."""
    # Open the image and convert to RGB
    img = Image.open(path).convert('RGB')
    
    # Get image dimensions
    width, height = img.size

    # Determine the size of the square crop
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image to a centered square
    img = img.crop((left, top, right, bottom))
    
    # Resize the cropped image to the target resolution using LANCZOS resampling
    img = img.resize(resize_to, Image.Resampling.LANCZOS)
    
    # Convert the image to a tensor (shape: C x H x W, normalized to [0, 1])
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    return img_tensor

def get_filename_from_path(path):
    # return file name without extension
    fname = os.path.basename(path)
    fname = os.path.splitext(fname)[0]
    return fname

def find_available_filename(folder, prefix):
    i = 0
    filename = f'{prefix}_{i:03d}'
    os.path.isdir(os.path.join(folder, filename))
    while os.path.isdir(os.path.join(folder, filename)):
        i += 1
        filename = f'{prefix}_{i:03d}'
    return filename

def crop_to_multiple(x, m):
    H, W = x.shape[-2:]
    x_cropped = x[:,:, :H - H % m, :W - W % m]
    return x_cropped

def _get_x_init(y_guidance, forward_model, transpose_operator, mask, cfg):
        """
        Compute the initial estimate for x using the transpose model.

        Args:
            y_guidance (torch.Tensor): The observed low-resolution image.
            forward_model: The forward operator with transpose support.

        Returns:
            torch.Tensor: Initial estimate of x.
        """
        if cfg.problem.type == 'inpainting_squared_mask':
            # Get tensor dimensions
            B, C, H, W = y_guidance.shape  # [1, 3, 1024, 1024]

            xinit = torch.clone(y_guidance)
            kernel_size = 19  # Moving average kernel size

            # Ensure mask is binary
            mask = (mask < 0.5).float()  # Ensure mask is 0 or 1

            # Create dilation kernel for border extraction
            dilation_kernel_size = 3  # Controls border thickness
            dilation_kernel = torch.ones((1, 1, dilation_kernel_size, dilation_kernel_size), device=xinit.device)

            # Dilate the mask to get border region
            dilated_mask = F.conv2d(mask, dilation_kernel, padding=dilation_kernel_size // 2)
            dilated_mask = (dilated_mask > 0).float()

            # Create the averaging kernel for filling (per channel)
            avg_kernel = torch.ones((C, 1, kernel_size, kernel_size), device=xinit.device) / (kernel_size ** 2)

            # Iterative filling process
            for _ in range(200):  # Number of iterations (tune as needed)
                # Apply convolution to average nearby pixels
                blurred = F.conv2d(xinit, avg_kernel, padding=kernel_size // 2, groups=C)

                # Identify pixels to fill: masked area minus already filled pixels
                fill_mask = mask.repeat(1, C, 1, 1)

                # Apply blurred values to the masked area
                xinit = xinit * (1 - fill_mask) + blurred * fill_mask

                # Break if all masked pixels are filled
                if fill_mask.sum() == 0:
                    break

            # Apply the filled area back to y_guidance
            y_guidance = xinit
        else:
            # Step 1: Solve A A^T u = y_guidance  for u
            # This requires a solver, e.g., conjugate gradients

            def A(x):
                return forward_model(x)

            def AT(x):
                return transpose_operator(x)

            def ATA(x):
                return AT(A(x))

            # Now solve: ATA(u) = AT(y_guidance)

            # CG iteration:
            b = AT(y_guidance.clone().detach())
            b.requires_grad = False
            
            u = dinv.optim.utils.conjugate_gradient(
                ATA, 
                b, 
                max_iter=1e2,
                tol=1e-5,
                eps=1e-8,
                verbose=False
            )

            # Step 2: Compute x_init = A^T u
            xinit = b.clip(-1, 1)
            
        return xinit, y_guidance