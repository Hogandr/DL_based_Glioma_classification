# DL_based_Glioma_classification
This repository contains the code for the master's thesis "A Comparative Study of Fully Automated Deep Learning Approaches for Predicting IDH Mutation Status of Glioma Patients Using 3D PET Imaging".

## Repository Structure

- **[Preprocessing](./Preprocessing)**  
  Contains script to run cropping of medical images.

- **[Supervised](./Supervised)**  
  Contains scripts to run supervised deep learning models.

- **[Self-supervised](./Self-supervised)**  
  Contains scripts to pretrain a model using self-supervised learning and to train a random forest classifier on the learned representations.

- **[Yaml_Generator.ipynb](./Yaml_Generator.ipynb)**  
  Jupyter notebook to generate YAML configuration files for different training approaches.  
  ➤ The user must define the paths and settings for each setup manually.

- **[requirements.txt](./requirements.txt)**  
  ➤ Developed with Python version `3.10.12`  
  ➤ Install all required dependencies using:  
  ```bash
  pip install -r requirements.txt
