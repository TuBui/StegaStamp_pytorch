import torch
import numpy as np
import skimage.metrics
import lpips

def normalise(x):
    """
    Normalise image array to range [-1, 1] and tensor.
    Args:
        x (np.ndarray): Image array of shape (N, H, W, C) in range [0, 255].
    Returns:
        (torch.Tensor): Image tensor of shape (N, C, H, W) in range [-1, 1].
    """
    x = x.astype(np.float32)
    x = x / 255
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x)
    x = x.permute(0, 3, 1, 2)
    return x


def unormalise(x, vrange=[-1, 1]):
    """
    Unormalise image tensor to range [0, 255] and RGB array.
    Args:
        x (torch.Tensor): Image tensor of shape (N, C, H, W) in range [-1, 1].
    Returns:
        (np.ndarray): Image array of shape (N, H, W, C) in range [0, 255].    
    """
    x = (x - vrange[0])/(vrange[1] - vrange[0])
    x = x * 255
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().numpy().astype(np.uint8)
    return x


def compute_mse(x, y):
    """
    Compute mean squared error between two image arrays.
    Args:
        x (np.ndarray): Image of shape (N, H, W, C) in range [0, 255].
        y (np.ndarray): Image of shape (N, H, W, C) in range [0, 255].
    Returns:
        (1darray): Mean squared error.
    """
    return np.square(x - y).reshape(x.shape[0], -1).mean(axis=1)


def compute_psnr(x, y):
    """
    Compute peak signal-to-noise ratio between two images.
    Args:
        x (np.ndarray): Image of shape (N, H, W, C) in range [0, 255].
        y (np.ndarray): Image of shape (N, H, W, C) in range [0, 255].
    Returns:
        (float): Peak signal-to-noise ratio.
    """
    return 10 * np.log10(255 ** 2 / compute_mse(x, y))


def compute_ssim(x, y):
    """
    Compute structural similarity index between two images.
    Args:
        x (np.ndarray): Image of shape (N, H, W, C) in range [0, 255].
        y (np.ndarray): Image of shape (N, H, W, C) in range [0, 255].
    Returns:
        (float): Structural similarity index.
    """
    return np.array([skimage.metrics.structural_similarity(xi, yi, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255) for xi, yi in zip(x, y)])


def compute_lpips(x, y, net='alex'):
    """
    Compute LPIPS between two images.
    Args:
        x (torch.Tensor): Image tensor of shape (N, C, H, W) in range [-1, 1].
        y (torch.Tensor): Image tensor of shape (N, C, H, W) in range [-1, 1].
    Returns:
        (float): LPIPS.
    """
    lpips_fn = lpips.LPIPS(net=net, verbose=False).cuda() if isinstance(net, str) else net
    x, y = x.cuda(), y.cuda()
    return lpips_fn(x, y).detach().cpu().numpy().squeeze()