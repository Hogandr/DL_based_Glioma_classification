# Self-supervised

This folder contains two Python scripts: one for training a self-supervised deep learning model, and the second for using the learned representations for downstream classification tasks.

## Contents

1. **[U_Net.py](./U_Net.py)**  
   Run a self-supervised U-Net based on the [Models Genesis](https://github.com/MrGiovanni/SuPreM/tree/main?tab=readme-ov-file) pretext tasks.

2. **[Random_Forest.py](./Random_Forest.py)**  
   Train a random forest classifier using features extracted from the self-supervised U-Net (Models Genesis).

## Running on SLURM

To run the models on a SLURM cluster, use the provided shell scripts:

- **[U_Net.sh](./U_Net.sh)**
- **[Random_Forest.sh](./Random_Forest.sh)**

> *Make sure to modify the scripts to match your system’s paths and resource requirements.*

## Input Format

- **Image Data**: NIfTI format (`.nii` / `.nii.gz`)  
- **Tabular Data**: Paths to image files and corresponding tabular data should be defined in Excel files (e.g., [`Split_0.xlsx`](./Split_x.xlsx), `Split_1.xlsx`, `Split_2.xlsx`) representing different data splits.
