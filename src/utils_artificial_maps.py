import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def create_julia_fractal(height: int = 256, width: int = 128, scale: float = 0.9, radius: int = 2, max_iterations: int = 255):
    """
    Create a Julia set fractal.

    Args:
        height (int): Height of the fractal image. Default is 256.
        width (int): Width of the fractal image. Default is 128.
        scale (float): Scaling factor for the fractal. Must be less than 1.0. Default is 0.9.
        radius (int): Escape radius for the fractal. Default is 2.
        max_iterations (int): Maximum number of iterations. Default is 255.

    Returns:
        np.ndarray: Normalized fractal image as a 2D numpy array.
    """
    assert scale < 1.0, "Scale must be less than 1.0"

    # Generate a random complex number for the Julia set
    f = np.random.uniform(-1, 1) + 1j * np.random.uniform(-0.5, 0.5)

    # Calculate the scaling factor
    s = min(height, width) * scale

    # Create the complex plane
    x = np.linspace(-width/s, width/s, num=width)
    y = np.linspace(-height/s, height/s, num=height)
    Z = x[None, :] + 1j * y[:, None]

    # Initialize the mask and iteration count arrays
    M = np.ones((height, width), dtype=bool)
    N = np.zeros((height, width), dtype=int)
    C = np.full((height, width), f)

    # Iterate to create the fractal
    for i in range(max_iterations):
        Z[M] = Z[M] * Z[M] + C[M]
        M[np.abs(Z) > radius] = False
        N[M] = i

    # Normalize the result
    return N / np.max(N)


def center_crop(img: np.ndarray, bounding: int) -> np.ndarray:
    """
    Perform a center crop on an image.

    Args:
        img (np.ndarray): Input image as a 2D numpy array.
        bounding (int): Size of the square crop.

    Returns:
        np.ndarray: Cropped image as a 2D numpy array.

    Raises:
        ValueError: If the bounding size is larger than the image dimensions.
    """
    # Get image dimensions
    height, width = img.shape

    # Check if crop size is valid
    if bounding > min(height, width):
        raise ValueError("Bounding size must not be larger than the image dimensions.")

    # Calculate start indices for cropping
    start_y = (height - bounding) // 2
    start_x = (width - bounding) // 2

    # Return the cropped image
    return img[start_y:start_y+bounding, start_x:start_x+bounding]


def swap_channels(img, t=0.5, p=0.01):
    """
    Randomly swaps the channels of some pixels in an image.

    Args:
        img (numpy.ndarray): The input image with shape (height, width, channels).
        t (float): Threshold value to select pixels for channel swapping.
        p (float): Proportion of the selected pixels to actually perform the swap on.

    Returns:
        numpy.ndarray: The image with some channels swapped.
    """
    # Find the coordinates of pixels where the intensity is greater than the threshold t
    targets = np.argwhere(img > t)

    # Sample a proportion p of the targets
    num_samples = int(len(targets) * p)
    samples = random.sample(list(map(tuple, targets)), num_samples)

    # Swap channels for the sampled pixels
    for s in samples:
        img[s[0], s[1]] = np.random.permutation(img[s[0], s[1]])

    return img


def creature_artificial_map(
        create_fake_cell: callable, 
        label: int, 
        num_features: int,
        height: int, 
        width: int, 
        num_slices: int = 3, 
        radius: int = 2,
        scale: float = 0.95, 
        is_normal: bool = False):

    if num_slices > 2:
        h, w = height//4, width//4
    else:
        h, w = height//3, width//3

    cell = create_fake_cell(height=h, width=w, scale=scale, radius=radius)
 
    if np.random.rand() > 0.5:
        bounding = min(cell.shape)
        cell = center_crop(cell, bounding)
        h, w = cell.shape

    cell *= (cell > 0.3)

    base_cell = np.repeat(cell, num_features).reshape((h, w, num_features))

    feature_map = np.zeros((height, width, num_features))
    offset = np.array([0, 0])
    offset[0] = feature_map.shape[0] // (num_slices+1)
    offset[1] = feature_map.shape[1] // (num_slices+1)
    for i in range(num_slices):        
        if is_normal:
            base_feature = np.random.uniform(0, 0.2, (h, w, num_features))
            feature_cell = F.softmax(torch.tensor(base_feature), dim=2).numpy()
            noise = np.random.uniform(0, 1, size=feature_cell.shape) > 0.9
            final_cell = torch.tensor(feature_cell * noise) * base_cell
        else:
            base_feature = np.random.uniform(-1, 0, (h, w, num_features))
            feature = np.random.uniform(0.8, 1, (h, w))
            base_feature[:, :, label] = feature
            feature_cell = F.softmax(torch.tensor(base_feature), dim=2).numpy()
            feature_cell = feature_cell * base_cell        
            noise = np.random.normal(0, 0.1, size=feature_cell.shape)
            swap_cell = swap_channels(feature_cell)
            final_cell = torch.tensor(swap_cell + noise)
            final_cell = F.softmax(final_cell, dim=2).numpy() * base_cell
        coord = offset*i + 50
        feature_map[coord[0]:coord[0] + h, coord[1]:coord[1] + w, :] = final_cell

    return feature_map, final_cell


