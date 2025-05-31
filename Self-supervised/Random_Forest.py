from Helper.config import models_genesis_config
from Helper.utils import *
from monai.data import DataLoader as MONAIDataLoader, ImageDataset
from monai.transforms import (
    Compose, RandSimulateLowResolution, RandGaussianNoise, RandAdjustContrast,
    RandRotate, RandAxisFlip, RandAffine, OneOf, EnsureChannelFirst, Resize
)
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, brier_score_loss, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Accuracy, MeanMetric, AUROC, F1Score
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAccuracy
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Dict, List
import argparse
import gc
import glob
import joblib
import monai
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import warnings
import yaml



class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 8 * (2 ** (depth+1)),act)
        layer2 = LUConv(8 * (2 ** (depth+1)), 8 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 8*(2**depth),act)
        layer2 = LUConv(8*(2**depth), 8*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class UNet3DWithClassification(nn.Module):
    def __init__(self, n_class=1, n_classes_global=1, act='relu'):
        super(UNet3DWithClassification, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(16,1,act)
        self.down_tr256 = DownTransition(32,2,act)
        self.down_tr512 = DownTransition(64,3,act)

        self.up_tr256 = UpTransition(128, 128,2,act)
        self.up_tr128 = UpTransition(64,64, 1,act)
        self.up_tr64 = UpTransition(32,32,0,act)
        self.out_tr = OutputTransition(16, n_class)

        self.global_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(128, n_classes_global)  # Fully Connected Layer

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        pooled_features = self.global_pool(self.out512)
        pooled_features = torch.flatten(pooled_features, start_dim=1)
        pooled_features_out = torch.flatten(self.out512, start_dim=1)
        classification_output = self.fc(pooled_features)

        return self.out, pooled_features_out


def get_images(dataframe, modality):
    """
    Extracts image file paths from a dataframe based on a given modality pattern.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image metadata, including 'Path'.
        modality (str): String pattern to match specific modality images (e.g., "T1", "FET", etc.).

    Returns:
        np.ndarray: Array of image file paths that match the modality pattern.
    """
    images = []

    # Optional debug: print patient column
    print(dataframe["Patients"])

    # Iterate over each row in the dataframe
    for index, row in dataframe.iterrows():
        # Construct the search pattern to find files of the given modality
        pattern = os.path.join(row['Path'], f"*{modality}*")

        # Find all matching files for this pattern
        matched = glob.glob(pattern)

        # Append found images to the overall list
        images.extend(matched)

    # Convert list to numpy array for consistency
    return np.array(images)


# ========================== Feature Extraction Function ==========================
def extract_features(data_loader, model, device):
    """
    Extracts high-level features from images using the global pooling layer of a U-Net model.

    Args:
        data_loader (DataLoader): DataLoader providing image-label batches.
        model (torch.nn.Module): Trained U-Net model with a classification head.
        device (torch.device): Device to run computations on (CPU or CUDA).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Extracted features and corresponding labels.
    """
    features = []
    labels = []

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(data_loader, leave=False, desc="Extracting Features"):
            # Free GPU memory
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            # Move inputs to the correct device
            x = x.to(device)
            y = y.cpu().numpy()  # Labels are not used on GPU

            # Forward pass through model to get classification features
            _, classification_output = model(x)

            # Store features and labels
            features.append(classification_output.cpu().numpy())
            labels.append(y)

    # Concatenate all batches into single numpy arrays
    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels


# ========================== Load Model Snapshot ==========================
def load_snapshot(model, path, rank, optimizer):
    """
    Loads a saved model snapshot from disk to resume training or for inference/feature extraction.

    Args:
        model (nn.Module): The model instance to load state into.
        path (str): Path to the saved snapshot (.pt file).
        rank (int): GPU rank to determine which device to load the model on.
        optimizer (Optimizer): The optimizer instance to restore its state.

    Returns:
        Tuple[nn.Module, int, Optimizer]: Loaded model, last epoch, and optimizer with restored state.
    """
    # Determine correct device (GPU or CPU)
    if torch.cuda.is_available():
        loc = f"cuda:{rank}"
    else:
        loc = torch.device('cpu')

    # Load snapshot dictionary from file
    snapshot = torch.load(path, map_location=loc)

    # Restore model and optimizer state
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot['OPTIMIZER_STATE_DICT'])

    # Retrieve the epoch the snapshot was saved at
    epoch = snapshot["EPOCHS_RUN"]
    print(f"✅ Loaded model snapshot from epoch {epoch}")

    return model, epoch, optimizer


# ========================== Feature Extraction Workflow ==========================
def extract_all_features(model, train_loader, val_loader, test_loader, device):
    """
    Wrapper function that runs feature extraction on train, validation, and test datasets.

    Args:
        model (nn.Module): Trained model with a classification head.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        test_loader (DataLoader): DataLoader for test set.
        device (torch.device): Device to run model inference on.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Combined train/val features and labels, test features and labels.
    """
    print("🚀 Extracting features for training...")
    train_features, train_labels = extract_features(train_loader, model, device)

    print("🚀 Extracting features for validation...")
    val_features, val_labels = extract_features(val_loader, model, device)

    # Combine training and validation features for downstream use (e.g., classifier training)
    print("🔗 Combining train and validation features...")
    combined_features = np.vstack((train_features, val_features))
    combined_labels = np.hstack((train_labels, val_labels))

    print("🚀 Extracting features for test...")
    test_features, test_labels = extract_features(test_loader, model, device)

    return combined_features, combined_labels, test_features, test_labels


# ========================== Random Forest Pipeline ==========================
def train_and_evaluate_random_forest(train_features, train_labels, test_features, test_labels, cfg):
    """
    Train a Random Forest classifier on extracted features and evaluate it on test data.

    Args:
        train_features (np.ndarray): Combined training features (e.g., from train + val sets).
        train_labels (np.ndarray): Labels corresponding to train_features.
        test_features (np.ndarray): Features from the test set.
        test_labels (np.ndarray): Labels from the test set.
        cfg (dict): Configuration dictionary, includes MODEL name for saving.

    Returns:
        None
    """

    print("🌲 Training Random Forest Classifier...")

    # Define a randomized parameter grid for hyperparameter optimization
    param_distributions = {
        'n_estimators': randint(50, 1000),  # Number of trees
        'bootstrap': [True, False],  # Use bootstrapping or not
        'max_features': uniform(0.005, 0.05),  # Fraction of features to consider at each split
    }
    print("Parameter distributions:", param_distributions)

    # Initialize Random Forest with class weighting for imbalanced data
    rf = RandomForestClassifier(random_state=0, class_weight="balanced")

    # Define stratified 5-fold cross-validation to preserve class distribution
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Perform hyperparameter tuning using randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of different combinations to try
        cv=stratified_kfold,
        n_jobs=-1,  # Use all available cores
        scoring='roc_auc',  # Optimize for AUC
        random_state=0,
        verbose=2
    )

    # Fit the RandomizedSearchCV with training data
    random_search.fit(train_features, train_labels)

    # Retrieve best estimator and save to disk
    best_rf = random_search.best_estimator_
    print("✅ Best Parameters Found:", random_search.best_params_)

    model_path = f"/data/core-nuk-physik/hlamprec/Experiments/U_NET_TRANSFER/TRANSFER_CLASS_MODELS/RF_{cfg['MODEL']}_Config={CONFIG}_SPLIT={SPLIT}.pkl"
    joblib.dump(best_rf, model_path)

    # Evaluate the best model on test set
    print("🧪 Evaluating on test data...")
    test_predictions = best_rf.predict(test_features)
    test_probabilities = best_rf.predict_proba(test_features)[:, 1]

    # Compute evaluation metrics
    brier = brier_score_loss(test_labels, test_predictions)
    accuracy = accuracy_score(test_labels, test_predictions)
    auc = roc_auc_score(test_labels, test_probabilities)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)

    # Display metrics
    print("\n📊 Test Set Performance:")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"AUC             : {auc:.4f}")
    print(f"Brier Score     : {brier:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    print("\n📝 Classification Report:")
    print(classification_report(test_labels, test_predictions))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))


# ========================== Main Workflow ==========================
if __name__ == "__main__":
    # -----------------------------------------------
    # Parse command-line argument for config ID
    # -----------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, required=False, help="Additional config ID.")
    args = parser.parse_args()
    SPLIT = args.config_id

    # -----------------------------------------------
    # Set up basic environment and config
    # -----------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = 0              # For DDP: local process rank (assuming single-GPU setup)
    world_size = 1        # For DDP: number of processes

    # Define model and configuration ID
    CONFIG = 0
    MODEL = "X"
    print("CONFIG")
    print(CONFIG)

    # Path to model checkpoint
    snapshot_path = f"/archive1/core-nuk-physik/hlamprec/Results/{MODEL}/MODELS/{MODEL}_Config={CONFIG}_SPLIT={SPLIT}.pt"

    # Define config dictionary
    cfg = {
        "SPLIT": SPLIT,
        "modality": "brats_normFET20-40_cropped.",
        "batch_size": 4,
        "WORKERS_PER_GPU": 2,
        "SEED": 42,
        "path_splits": "/data/core-nuk-physik/hlamprec/Information/New/HR+_no_missing_balanced_train_val_test_split",
        "snapshot_path": snapshot_path,
        "MODEL": MODEL,
    }

    # -----------------------------------------------
    # Load model and optimizer state
    # -----------------------------------------------
    model = UNet3DWithClassification(n_class=1, n_classes_global=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model, _, optimizer = load_snapshot(model, cfg["snapshot_path"], rank, optimizer)

    # -----------------------------------------------
    # Load and prepare data from Excel
    # -----------------------------------------------
    path_splits = cfg["path_splits"] + "_" + str(cfg["SPLIT"]) + ".xlsx"
    dataframe_train = pd.read_excel(path_splits, sheet_name='train')
    dataframe_validation = pd.read_excel(path_splits, sheet_name='validation')
    dataframe_test = pd.read_excel(path_splits, sheet_name='test')

    train_images = get_images(dataframe_train, cfg["modality"])
    validation_images = get_images(dataframe_validation, cfg["modality"])
    test_images = get_images(dataframe_test, cfg["modality"])

    train_labels = np.array(dataframe_train["Status"].tolist(), dtype=np.float64)
    validation_labels = np.array(dataframe_validation["Status"].tolist(), dtype=np.float64)
    test_labels = np.array(dataframe_test["Status"].tolist(), dtype=np.float64)

    # -----------------------------------------------
    # Define MONAI transforms and datasets
    # -----------------------------------------------
    transform = Compose([EnsureChannelFirst(), Resize((80, 80, 80))])
    train_dataset = ImageDataset(train_images, train_labels, transform=transform)
    validation_dataset = ImageDataset(validation_images, validation_labels, transform=transform)
    test_dataset = ImageDataset(test_images, test_labels, transform=transform)

    # -----------------------------------------------
    # Distributed Samplers and Dataloaders
    # -----------------------------------------------
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=cfg["SEED"])
    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['WORKERS_PER_GPU'],
                              sampler=train_sampler, persistent_workers=True, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=4, num_workers=cfg['WORKERS_PER_GPU'],
                                   sampler=validation_sampler, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=cfg['WORKERS_PER_GPU'],
                             sampler=test_sampler, persistent_workers=True, pin_memory=True)

    # -----------------------------------------------
    # Extract features from model for each dataset
    # -----------------------------------------------
    combined_features, combined_labels, test_features, test_labels = extract_all_features(
        model, train_loader, validation_loader, test_loader, device
    )

    # -----------------------------------------------
    # Train and evaluate a Random Forest classifier
    # -----------------------------------------------
    train_and_evaluate_random_forest(combined_features, combined_labels, test_features, test_labels, cfg)
