import math
import os
import random
import copy
import scipy
import string
import numpy as np
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from torch import nn
import torch


def bernstein_poly(i, n, t):
    """
    Compute the Bernstein polynomial value for index `i` and order `n` at parameter `t`.

    Parameters:
    - i (int): Polynomial term index.
    - n (int): Polynomial degree.
    - t (float or np.ndarray): Time or interpolation parameter in [0, 1].

    Returns:
    - float or np.ndarray: Value(s) of the Bernstein polynomial.
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
    Generate a Bezier curve from a set of control points.

    Parameters:
    - points (list of lists/tuples): Control points defining the curve (e.g., [[x0, y0], [x1, y1], ..., [xn, yn]]).
    - nTimes (int): Number of samples along the curve.

    Returns:
    - Tuple of np.ndarray: x and y coordinates of the Bezier curve.

    Reference:
    - http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)

    # Calculate Bernstein polynomial values for all control points
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(nPoints)])

    # Generate Bezier curve coordinates
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    """
    Perform random data augmentation via axis flipping.

    Parameters:
    - x (np.ndarray): Input data (e.g., image volume).
    - y (np.ndarray): Corresponding label or target.
    - prob (float): Probability to apply a flip per iteration (default: 0.5).

    Returns:
    - Tuple: Augmented x and y.
    """
    cnt = 3  # Limit the number of flip attempts
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])  # Random axis for flipping
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt -= 1

    return x, y


def nonlinear_transformation(x, prob=0.5, max_y=1):
    """
    Apply a nonlinear transformation to input `x` using a random Bezier curve.

    Parameters:
    - x (np.ndarray): Input data (e.g., intensity values).
    - prob (float): Probability of applying the transformation.
    - max_y (float): Maximum y-value for Bezier control points.

    Returns:
    - np.ndarray: Nonlinearly transformed input.
    """
    if random.random() >= prob:
        return x  # Return unmodified if probability not met

    # Randomly generate 4 control points for the Bezier curve
    points = [[0, 0],
              [random.random() * max_y, random.random() * max_y],
              [random.random() * max_y, random.random() * max_y],
              [max_y, max_y]]

    # Decompose control points
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    # Generate the Bezier curve
    xvals, yvals = bezier_curve(points, nTimes=100000)

    # Optionally sort x/y values to ensure monotonicity
    if random.random() < 0.5:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)

    # Interpolate `x` through the generated nonlinear curve
    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    """
    Apply local pixel shuffling within small random blocks of the image volume.

    Parameters:
    - x (np.ndarray): Input 4D image volume of shape (1, D, H, W).
    - prob (float): Probability to apply the transformation (default: 0.5).

    Returns:
    - np.ndarray: Shuffled image volume.
    """
    if random.random() >= prob:
        return x  # Skip augmentation

    image_temp = copy.deepcopy(x)  # Output image
    orig_image = copy.deepcopy(x)  # Reference copy
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000  # Number of local shuffle regions

    for _ in range(num_block):
        # Define random block size
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)

        # Random starting coordinates
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)

        # Extract block and shuffle its pixels
        window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                               noise_y:noise_y + block_noise_size_y,
                               noise_z:noise_z + block_noise_size_z]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, block_noise_size_y, block_noise_size_z))

        # Replace block in output image
        image_temp[0, noise_x:noise_x + block_noise_size_x,
                      noise_y:noise_y + block_noise_size_y,
                      noise_z:noise_z + block_noise_size_z] = window

    return image_temp


def image_in_painting(x):
    """
    Simulate image corruption by overwriting random cuboid patches with noise.

    Parameters:
    - x (np.ndarray): Input 4D image volume of shape (1, D, H, W).

    Returns:
    - np.ndarray: Image with random in-painted patches.
    """
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5  # Limit number of in-painting attempts

    while cnt > 0 and random.random() < 0.95:
        # Random block size
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)

        # Random block location
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)

        # Replace region with random noise
        x[:, noise_x:noise_x + block_noise_size_x,
              noise_y:noise_y + block_noise_size_y,
              noise_z:noise_z + block_noise_size_z] = np.random.rand(
                  block_noise_size_x, block_noise_size_y, block_noise_size_z
              )

        cnt -= 1

    return x


def image_out_painting(x):
    """
    Simulate missing context by keeping only random cuboid patches and replacing the rest with noise.

    Parameters:
    - x (np.ndarray): Input 4D image volume of shape (1, D, H, W).

    Returns:
    - np.ndarray: Out-painted image.
    """
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)

    # Start with full noise
    x = np.random.rand(*x.shape)

    # Generate a large central cuboid to preserve from original image
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)

    # Insert preserved patch into noisy background
    x[:, noise_x:noise_x + block_noise_size_x,
         noise_y:noise_y + block_noise_size_y,
         noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                            noise_y:noise_y + block_noise_size_y,
                                                            noise_z:noise_z + block_noise_size_z]

    # Optionally insert a few more patches
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)

        x[:, noise_x:noise_x + block_noise_size_x,
             noise_y:noise_y + block_noise_size_y,
             noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                                noise_y:noise_y + block_noise_size_y,
                                                                noise_z:noise_z + block_noise_size_z]
        cnt -= 1

    return x


def find_nonzero_bounding_box(volume):
    """
    Identifies the smallest axis-aligned bounding box that contains all non-zero values
    within a given 3D volume. The bounding box is symmetrically expanded to ensure
    a minimum size and clamped to remain within volume boundaries.

    Parameters:
    - volume (np.ndarray): A 4D volume with shape (1, D, H, W).

    Returns:
    - Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
      Minimum and maximum coordinates of the bounding box (inclusive).
    """
    volume_bounding = copy.deepcopy(volume)[0]
    print("find_nonzero_bounding_box")
    print(volume_bounding.shape)

    # Get coordinates of all non-zero voxels
    non_zero_indices = np.array(np.nonzero(volume_bounding))
    min_coords = np.min(non_zero_indices, axis=1)
    max_coords = np.max(non_zero_indices, axis=1)

    # Calculate the current size and ensure a minimum size
    current_size = max_coords - min_coords + 1
    max_side = max(np.max(current_size), 15)

    # Center the bounding box around the midpoint
    midpoint = (min_coords + max_coords) // 2
    half_size = max_side // 2
    min_coords = np.maximum(midpoint - half_size, 0)
    max_coords = np.minimum(min_coords + max_side - 1, np.array(volume_bounding.shape) - 1)

    return tuple(min_coords), tuple(max_coords)


def extract_and_transform_subvolume(volume_x, volume_y, transformation_fn, **kwargs):
    """
    Extracts a 3D subvolume from the input volumes based on the non-zero region in volume_x,
    applies a transformation to both x and y subvolumes, and reinserts the transformed data
    back into the original volume locations.

    Parameters:
    - volume_x (np.ndarray): 4D input volume with shape (1, D, H, W).
    - volume_y (np.ndarray): 4D target volume (same shape as volume_x).
    - transformation_fn (callable): Function to apply on the extracted subvolumes.
    - **kwargs: Optional keyword arguments passed to the transformation function.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The modified input and target volumes.
    """
    # Determine the non-zero region in volume_x
    min_coords, max_coords = find_nonzero_bounding_box(volume_x)
    print(min_coords)

    # Extract corresponding subvolumes
    subvolume_x = volume_x[:, min_coords[0]:max_coords[0] + 1,
                  min_coords[1]:max_coords[1] + 1,
                  min_coords[2]:max_coords[2] + 1]

    subvolume_y = volume_y[:, min_coords[0]:max_coords[0] + 1,
                  min_coords[1]:max_coords[1] + 1,
                  min_coords[2]:max_coords[2] + 1]

    print("extract_and_transform_subvolume")
    print(subvolume_x.shape)
    print(subvolume_y.shape)

    # Apply transformation to subvolumes
    transformed_subvolume_x, transformed_subvolume_y = transformation_fn(
        subvolume_x, subvolume_y, **kwargs
    )

    # Reinsert transformed subvolumes back into the original volumes
    volume_x[:, min_coords[0]:max_coords[0] + 1,
    min_coords[1]:max_coords[1] + 1,
    min_coords[2]:max_coords[2] + 1] = transformed_subvolume_x

    volume_y[:, min_coords[0]:max_coords[0] + 1,
    min_coords[1]:max_coords[1] + 1,
    min_coords[2]:max_coords[2] + 1] = transformed_subvolume_y

    return volume_x, volume_y


def generate_pair(img, config):
    """
    Generates training pairs (x, y) for self-supervised learning from 3D images.

    For each image, a copy is made and a set of data augmentation and transformation
    operations is applied. These modified volumes (x) serve as inputs, while the
    original volumes (y) serve as targets for reconstruction.

    Parameters:
    - img (np.ndarray): Input batch of shape (B, C, D, H, W).
    - config (Namespace): Configuration object containing transformation probabilities.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Transformed images (x) and their original versions (y).
    """
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]

    # Generate index list for all images in the batch
    index = [i for i in range(img.shape[0])]
    y = img[index]  # Original target images
    x = copy.deepcopy(y)  # Deep copy for transformation

    # Apply transformations to each image individually
    for n in range(len(index)):
        # Duplicate original image to apply augmentation
        x[n] = copy.deepcopy(y[n])

        # Extract a meaningful subvolume and apply transformations
        x[n], y[n] = extract_and_transform_subvolume(
            x[n], y[n],
            transformation_fn=apply_transformations,
            config=config
        )

    return x, y


def apply_transformations(subvolume_x, subvolume_y, config):
    """
    Applies a series of data augmentations to the extracted subvolumes.

    Some transformations (e.g., flipping) are applied to both x and y.
    Others (e.g., pixel shuffling, non-linear distortions, in/out-painting)
    are applied only to x to create a self-supervised learning signal.

    Parameters:
    - subvolume_x (np.ndarray): Input subvolume to be transformed.
    - subvolume_y (np.ndarray): Ground truth subvolume (typically unchanged).
    - config (Namespace): Configuration with augmentation parameters.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Transformed input and unchanged ground truth subvolumes.
    """

    # Flip both input and label volumes with a given probability
    subvolume_x, subvolume_y = data_augmentation(subvolume_x, subvolume_y, config.flip_rate)

    # Locally shuffle voxels in the input volume only
    subvolume_x = local_pixel_shuffling(subvolume_x, prob=config.local_rate)

    # Apply non-linear deformation to the input volume
    max_y = subvolume_y.max()  # Use max value for proper scaling
    subvolume_x = nonlinear_transformation(subvolume_x, config.nonlinear_rate, max_y)

    # Randomly apply either inpainting or outpainting
    if random.random() < config.paint_rate:
        if random.random() < config.inpaint_rate:
            subvolume_x = image_in_painting(subvolume_x)
        else:
            subvolume_x = image_out_painting(subvolume_x)

    return subvolume_x, subvolume_y

class WeightedLossWithNonzeroRatio(nn.Module):
    """
    Custom loss function that reweights the loss based on the ratio of non-zero pixels
    in the input volume. This is useful in sparse data settings (e.g. medical imaging)
    where most voxels may be background (zero).

    Parameters:
    - loss_func (nn.Module): Base loss function (e.g., MSELoss) with `reduction='none'`.
    - total_pixels (int): Total number of pixels per sample used for normalization.
    - fixed (bool): If True, uses the fixed total_pixels value; otherwise, calculates dynamically.
    """
    def __init__(self, loss_func=nn.MSELoss(reduction='none'), total_pixels=231195, fixed=True):
        super(WeightedLossWithNonzeroRatio, self).__init__()
        self.loss_func = loss_func
        self.total_pixels = total_pixels
        self.fixed = fixed

    def forward(self, y_hat, x):
        # Compute element-wise loss
        loss_per_element = self.loss_func(y_hat, x)

        # Compute mean loss per sample
        batch_size = y_hat.size(0)
        loss_per_sample = loss_per_element.view(batch_size, -1).mean(dim=1)

        # Count non-zero voxels per sample
        nonzero_counts = (x != 0).sum(dim=(1, 2, 3, 4))

        # Total number of pixels per sample
        total_pixels = self.total_pixels if self.fixed else x[0].numel()

        # Compute non-zero ratio and apply minimum threshold
        nonzero_ratios = nonzero_counts / total_pixels
        nonzero_ratios[nonzero_ratios <= 0.05] = 0.05

        # Normalize the loss by the non-zero ratio
        loss_per_sample_normalized = loss_per_sample / nonzero_ratios

        # Return the mean loss across the batch
        return loss_per_sample_normalized.mean()


class WeightedLossWithNonzeroRatio_ROI(nn.Module):
    """
    ROI-aware custom loss function that gives higher weight to non-zero (i.e., region of interest)
    voxels. Combines base loss with a weighted mask emphasizing ROI areas.

    Parameters:
    - loss_func (nn.Module): Base loss function (e.g., MSELoss) with `reduction='none'`.
    - total_pixels (int): Total number of pixels per sample used for normalization.
    - fixed (bool): If True, uses the fixed total_pixels value; otherwise, calculates dynamically.
    """
    def __init__(self, loss_func=nn.MSELoss(reduction='none'), total_pixels=231195, fixed=True):
        super(WeightedLossWithNonzeroRatio_ROI, self).__init__()
        self.loss_func = loss_func
        self.total_pixels = total_pixels
        self.fixed = fixed

    def forward(self, y_hat, x):
        # Create a binary mask for the ROI (non-zero areas)
        mask = (x != 0).float()

        # Combine unweighted and ROI-weighted losses (10% background, 90% ROI)
        loss_per_element = self.loss_func(y_hat, x) * 0.1 + self.loss_func(y_hat, x) * mask * 0.9

        # Compute mean loss per sample
        batch_size = y_hat.size(0)
        loss_per_sample = loss_per_element.view(batch_size, -1).mean(dim=1)

        # Count non-zero voxels per sample
        nonzero_counts = (x != 0).sum(dim=(1, 2, 3, 4))

        # Total number of pixels per sample
        total_pixels = self.total_pixels if self.fixed else x[0].numel()

        # Compute non-zero ratio and apply minimum threshold
        nonzero_ratios = nonzero_counts / total_pixels
        nonzero_ratios[nonzero_ratios <= 0.05] = 0.05

        # Normalize the loss by the non-zero ratio
        loss_per_sample_normalized = loss_per_sample / nonzero_ratios

        # Return the mean loss across the batch
        return loss_per_sample_normalized.mean()
