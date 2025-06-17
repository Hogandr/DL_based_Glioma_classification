# Preprocessing

This folder contains a Python script for cropping medical images based on tumor segmentations.

## Contents

1. **[Cropping.py](./Cropping.py)**  
    Crop the tumor region from PET or MRI images using a pair of segmentation masks.  
    ➤ Supports optional fixed-size cropping for uniform input dimensions across all patients.
    

## Input Format

- **Image Data**: NIfTI format (`.nii` / `.nii.gz`)
    
- **Tabular Data**: Patient list specified in an Excel file (e.g., [`Patients.xlsx`](./Patients.xlsx))
    

The Excel file should contain a single column with patient IDs (e.g., `GLIO_0001`, `GLIO_0002`, ...), indexed as the first column.

## Folder Structure

```
├── data/
│   ├── segmentations/               # Input segmentations (used to localize ROI)
│   │   ├── GLIO_0000/
│   │   │   └── results/
│   │   │       ├── segmentation_1.nii.gz
│   │   │       └── segmentation_2.nii.gz
│   │   └── ...
│   │
│   ├── images/                      # Input images to be cropped
│   │   ├── GLIO_0000/
│   │   │   └── results/
│   │   │       └── Image_to_crop.nii.gz
│   │   └── ...
│   │
│   └── Patients.xlsx                # Excel file with list of patient IDs
```

> _Ensure that segmentation and image files follow consistent naming conventions to match the script’s pattern-based loading._
