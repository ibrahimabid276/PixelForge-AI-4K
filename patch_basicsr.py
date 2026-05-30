import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Patch basicsr before importing realesrgan
import basicsr.data.degradations as degradations
import basicsr.utils as utils

# ============================================================
# Patch degradations module
# ============================================================

def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """Create a circular low-pass kernel."""
    sigma2 = cutoff ** 2
    mid = kernel_size // 2
    k = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            r2 = (i - mid) ** 2 + (j - mid) ** 2
            k[i, j] = np.exp(-r2 / (2 * sigma2))
    k = k / k.sum()
    return k


def random_add_gaussian_noise_pt(img, sigma_range=(0.1, 10), clip=True, rounds=False, device='cuda'):
    """Add Gaussian noise to PyTorch tensor."""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = torch.randn_like(img) * sigma / 255.0
    out = img + noise
    if clip:
        out = out.clamp(0, 1)
    if rounds:
        out = (out * 255.0).round() / 255.0
    return out


def random_add_poisson_noise_pt(img, scale_range=(0.1, 10), clip=True, rounds=False, device='cuda'):
    """Add Poisson noise to PyTorch tensor."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    img = img * 255.0
    noise = torch.poisson(img * scale) / scale - img
    out = img + noise
    if clip:
        out = out.clamp(0, 255)
    if rounds:
        out = out.round()
    return out / 255.0


# Inject the missing functions into degradations module
degradations.circular_lowpass_kernel = circular_lowpass_kernel
degradations.random_add_gaussian_noise_pt = random_add_gaussian_noise_pt
degradations.random_add_poisson_noise_pt = random_add_poisson_noise_pt


# ============================================================
# Patch utils module - Add missing classes
# ============================================================

class DiffJPEG(nn.Module):
    """Simulate JPEG compression (simplified version)."""
    def __init__(self, differentiable=False):
        super().__init__()
        self.differentiable = differentiable
    
    def forward(self, x, quality=None):
        # Simplified: just return the input
        # A full implementation would simulate JPEG compression
        return x


class USMSharp(nn.Module):
    """Unsharp Masking for image sharpening."""
    def __init__(self, radius=50, sigma=0):
        super().__init__()
        self.radius = radius
        self.sigma = sigma
    
    def forward(self, x):
        # Simplified: just return the input
        # A full implementation would apply unsharp masking
        return x


# Inject the missing classes into utils module
utils.DiffJPEG = DiffJPEG
utils.USMSharp = USMSharp


# ============================================================
# Patch missing img_process_util module
# ============================================================

def filter2D(img, kernel):
    """Apply 2D filter to image.
    
    Args:
        img: Input image tensor.
        kernel: Convolution kernel.
    
    Returns:
        Filtered image.
    """
    import torch.nn.functional as F
    
    # Ensure kernel is 4D
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution
    pad_h = kernel.size(2) // 2
    pad_w = kernel.size(3) // 2
    img = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    output = F.conv2d(img, kernel, padding=0)
    return output


# Create and inject the module
import types
img_process_util = types.ModuleType('basicsr.utils.img_process_util')
img_process_util.filter2D = filter2D
sys.modules['basicsr.utils.img_process_util'] = img_process_util


# ============================================================
# Patch download_util module
# ============================================================

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file from URL.
    
    Args:
        url: URL to download from.
        model_dir: Directory to save the file.
        progress: Show progress bar.
        file_name: File name.
    
    Returns:
        Path to downloaded file.
    """
    import urllib.request
    
    if model_dir is None:
        model_dir = os.path.join(os.path.expanduser('~'), '.cache', 'basicsr')
    
    os.makedirs(model_dir, exist_ok=True)
    
    if file_name is None:
        file_name = url.split('/')[-1]
    
    file_path = os.path.join(model_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {url} to {file_path}")
        urllib.request.urlretrieve(url, file_path)
    
    return file_path


# Import and patch download_util
import basicsr.utils.download_util as download_util
download_util.load_file_from_url = load_file_from_url

print("✓ Patched basicsr.degradations module")
print("✓ Patched basicsr.utils module")
print("✓ Patched basicsr.utils.img_process_util module")
print("✓ Patched basicsr.utils.download_util module")
