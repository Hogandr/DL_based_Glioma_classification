import os
import pandas as pd
import nibabel as nib
import numpy as np
import glob

def max_cropped_size(path_in_seg, path_excel, segmentation_to_use_1, segmentation_to_use_2):
    """
    Computes the maximum ROI (Region of Interest) dimensions across all patients
    based on the union of two segmentation volumes per patient.

    Args:
        path_in_seg (str): Base path to the segmentation results per patient.
        path_excel (str): Path to Excel file containing patient IDs.
        segmentation_to_use_1 (list): List of substrings identifying the first set of segmentations.
        segmentation_to_use_2 (list): List of substrings identifying the second set of segmentations.

    Returns:
        list: Maximum dimensions [y_max, x_max, z_max] across all ROIs.
    """
    # Combine segmentation identifiers
    segmentations = segmentation_to_use_1 + segmentation_to_use_2

    # Load patient IDs from Excel
    patients = pd.read_excel(path_excel, index_col=0).index.values
    print(patients)  # Optional: for debugging

    # Initialize maximum dimension trackers
    y_max, x_max, z_max = 0, 0, 0

    # Iterate over each patient
    for patient in patients:
        # Build path to results directory
        path_results = os.path.join(path_in_seg, patient, "results")
        segementations_path = []

        # Collect matching segmentation file paths for the patient
        for seg_name in segmentations:
            segementations_path.extend(glob.glob(os.path.join(path_results, f"*{seg_name}*")))

        print(patient)
        print(segementations_path)

        # Proceed only if exactly two segmentation files are found
        if len(segementations_path) == 2:
            # Load both segmentation volumes
            segmentation_1 = nib.load(segementations_path[0]).get_fdata()
            segmentation_2 = nib.load(segementations_path[1]).get_fdata()

            # Combine segmentations and find the ROI (non-zero voxels)
            combined_seg = segmentation_1 + segmentation_2
            ROI = np.argwhere(combined_seg)

            # Skip patients with no ROI
            if ROI.shape[0] == 0:
                print(f"Skipping patient {patient} due to empty ROI.")
                continue

            # Calculate ROI size in each dimension
            y, x, z = ROI.max(0) - ROI.min(0) + 1

            # Update global max if current dimensions are larger
            y_max = max(y_max, y)
            x_max = max(x_max, x)
            z_max = max(z_max, z)

            print("Dim:", [y, x, z])
            print("Max_Dim:", [y_max, x_max, z_max])

    # Return maximum dimensions found
    return [y_max, x_max, z_max]



def crop_ROI(path_in_seg, path_in_image, path_excel, segmentation_to_use_1, segmentation_to_use_2, image_to_crop,
             cropped_image_name, fixed_size, max_dim=[], path_out_image=[]):
    """
    Crops PET images based on the union of two segmentation masks, optionally using a fixed-size ROI.
    Applies zero-padding via embedding in a larger mask and saves the cropped result as a NIfTI file.

    Args:
        path_in_seg (str): Path to the segmentations directory.
        path_in_image (str): Path to the images directory.
        path_excel (str): Path to Excel file containing patient IDs.
        segmentation_to_use_1 (list): List of substrings for the first segmentation type.
        segmentation_to_use_2 (list): List of substrings for the second segmentation type.
        image_to_crop (list): List with one substring identifying the image to crop.
        cropped_image_name (str): Filename (suffix) for the saved cropped image.
        fixed_size (bool): If True, enforce a uniform ROI shape based on `max_dim`.
        max_dim (list): Dimensions [y, x, z] used when `fixed_size` is True.
        path_out_image (str): Base output path for saving cropped images.
    """
    # Combine all relevant file identifiers
    segmentation_and_image = segmentation_to_use_1 + segmentation_to_use_2 + image_to_crop

    # Load patient IDs from Excel
    patients = pd.read_excel(path_excel, index_col=0).index.values

    # Define the shape of the image and the zero-padded mask container
    object_dim = (240, 240, 155)
    mask_dim = (500, 500, 500)
    start_indices = [(mask_dim[i] - object_dim[i]) // 2 for i in range(3)]
    end_indices = [start_indices[i] + object_dim[i] for i in range(3)]

    for patient in patients:
        print(patient)

        # Construct paths to segmentation and image directories
        path_results_seg = os.path.join(path_in_seg, patient, "results")
        path_results_image = os.path.join(path_in_image, patient, "results")

        patient_images_path = []

        # Collect file paths for segmentations and PET image
        for i, image_identifier in enumerate(segmentation_and_image):
            if i == 2:  # Assuming the 3rd item is the image to crop
                patient_images_path.extend(
                    glob.glob(os.path.join(path_results_image, f"*{image_identifier}*"))
                )
            else:
                patient_images_path.extend(
                    glob.glob(os.path.join(path_results_seg, f"*{image_identifier}*"))
                )

        # Skip patient if fewer than 3 expected files are found
        if len(patient_images_path) != 3:
            print(f"Warning: Not enough images for this patient {patient}. Skipping this patient.")
            continue

        # Construct full output path and skip if file already exists
        temp_output_path = os.path.join(path_out_image, patient, "results", f"GLIO_{patient}_{cropped_image_name}.nii.gz")
        if os.path.exists(temp_output_path):
            print(f"Output file: {temp_output_path} already exists for patient {patient}. Skipping.")
            continue

        # Load the two segmentation masks and the corresponding image
        segmentation_1 = nib.load(patient_images_path[0])
        segmentation_2 = nib.load(patient_images_path[1])
        image = nib.load(patient_images_path[2])

        segmentation_raw_1 = segmentation_1.get_fdata()
        segmentation_raw_2 = segmentation_2.get_fdata()
        image_raw = image.get_fdata()

        # Combine the two segmentations into one mask
        segmentation_raw = segmentation_raw_1 + segmentation_raw_2

        # Initialize zero-padded mask volumes
        mask_image = np.zeros(mask_dim)
        mask_segmentation = np.zeros(mask_dim)

        # Embed original image and segmentation into the center of the mask
        mask_image[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
                   start_indices[2]:end_indices[2]] = image_raw
        mask_segmentation[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
                          start_indices[2]:end_indices[2]] = segmentation_raw

        print(mask_image.shape)
        print(mask_segmentation.shape)

        # Replace image and segmentation arrays with their padded versions
        image_raw = mask_image
        segmentation_raw = mask_segmentation

        # Zero out background (non-segmented) regions
        image_raw[segmentation_raw == 0] = 0

        # Extract the ROI (non-zero voxel coordinates)
        ROI = np.argwhere(segmentation_raw)

        if ROI.size == 0:
            print(f"Warning: ROI is an empty array for patient {patient}. Skipping this patient.")
            continue

        # Determine bounding box of the ROI
        (ystart, xstart, zstart), (ystop, xstop, zstop) = ROI.min(0), ROI.max(0) + 1

        # Apply fixed-size cropping if specified
        if fixed_size == True:
            epsilon = 1e-10
            # Define lower and upper bounds to match size of max_dim
            (ystart, xstart, zstart) = [ystart, xstart, zstart] - np.round(
                (max_dim - (ROI.max(0) - ROI.min(0) + 1)) / 2 + epsilon).astype(int)

            (ystop, xstop, zstop) = [ystop, xstop, zstop] + np.round(
                (max_dim - (ROI.max(0) - ROI.min(0) + 1)) / 2 - epsilon).astype(int)

        # Crop the image using the bounding box
        image_raw = image_raw[ystart:ystop, xstart:xstop, zstart:zstop]
        print([ystart, ystop, xstart, xstop, zstart, zstop])

        # Wrap cropped image in a NIfTI object
        stacked_img = nib.Nifti1Image(image_raw, image.affine)
        print(image.affine)
        print(stacked_img.shape)

        # Ensure output directory exists and save the cropped image
        os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
        nib.save(stacked_img, temp_output_path)

        print("Saved:", temp_output_path)




# Path to the directory containing patient segmentation results (e.g., output of a segmentation model)
path_in_seg = "/path/to/segmentations/"

# Segmentation file name patterns (e.g., model-specific outputs or manual annotations)
segmentation_to_use_1 = ["_segmentation_type_1"]
segmentation_to_use_2 = ["segmentation_type_2"]

# Path to Excel file listing patient identifiers (one per row, ideally as index column)
path_excel = "/path/to/patient_list.xlsx"

# Choose whether to automatically calculate max_dim from all ROIs or use a fixed value
use_auto_max_dim = True  # Set to False to manually define your own crop size

# Either compute max crop size from data or provide fixed dimensions manually
if use_auto_max_dim:
    max_dim = max_cropped_size(path_in_seg, path_excel, segmentation_to_use_1, segmentation_to_use_2)
else:
    max_dim = [0, 0, 0]  # Manually defined maximum crop dimensions (y, x, z)

# File name pattern of the image to crop (e.g., PET, CT, or SUV image)
image_to_crop = ["image_to_crop_pattern"]

# Desired name for the output cropped image (used in the output file name)
cropped_image_name = "cropped_image_output"

# Path to the directory containing the original full-size images (e.g., PET images)
path_in_image = "/path/to/images/"

# Output path where cropped images will be saved
path_out_image = "/path/to/output_crops/"

# === Run Cropping ===

crop_ROI(
    path_in_seg=path_in_seg,
    path_in_image=path_in_image,
    path_excel=path_excel,
    segmentation_to_use_1=segmentation_to_use_1,
    segmentation_to_use_2=segmentation_to_use_2,
    image_to_crop=image_to_crop,
    cropped_image_name=cropped_image_name,
    fixed_size=True,  # Set to False to crop to individual ROI size per patient
    max_dim=max_dim,
    path_out_image=path_out_image
)
