# Supervised

This folder contains various Python scripts for training supervised deep learning models using different input types and architectures.

## Contents

1. **[DenseNet_Singlechannel.py](./DenseNet_Singlechannel.py)**  
   Run DenseNet on a single-modality (single-channel) image input.  
   ➤ To use the custom CNN instead of DenseNet, refer to the model implementation in [`Helper_Custom.py`](./Helper/Helper_Custom.py).

2. **[DenseNet_Multichannel.py](./DenseNet_Multichannel.py)**  
   Run DenseNet on multi-channel/multi-modality image data.

3. **[DenseNet_Dual.py](./DenseNet_Dual.py)**  
   Run DenseNet using both image data and tabular data.

4. **[ResNet_Transfer.py](./ResNet_Transfer.py)**  
   Run a ResNet with transfer learned weights.  
   ➤ For Med3D, download pre-trained weights (resnet_10_23dataset) from the [MedicalNet repository](https://github.com/Tencent/MedicalNet).  
   ➤ For SuPreM, use the SegResNet weights available [here](https://github.com/MrGiovanni/SuPreM/tree/main?tab=readme-ov-file).

## Running on SLURM

To run the models on a SLURM cluster, use the provided shell script:

- **[Supervised.sh](./Supervised.sh)**

>  *Make sure to modify the script to match your system’s paths and resource requirements.*

## Input Format

- **Image Data**: NIfTI format (`.nii` / `.nii.gz`)
- **Tabular Data**: Paths to images and the tabular data should be specified in an Excel file (e.g., [`Split_0.xlsx`](./Split_x.xlsx), `Split_1.xlsx`, `Split_2.xlsx`) representing different data splits.
