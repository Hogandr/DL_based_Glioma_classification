import argparse
import os
import warnings
import random
import glob
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import pandas as pd
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Accuracy, MeanMetric, AUROC, F1Score
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAccuracy
from tqdm import tqdm
import monai
from monai.data import ImageDataset
from monai.transforms import (
    Compose, RandGaussianNoise, RandRotate, RandAxisFlip, RandAffine,
    OneOf, EnsureChannelFirst
)
from Helper.Helper_Transfer import *

# Set environment variable to control CUDA memory allocation behavior in PyTorch
# This limits the maximum split size for memory allocations to 128 MB, which can help with fragmentation issues.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Suppress specific warnings related to PyTorch's multiprocessing and tensor copying.
# These can clutter the output especially in legacy or multi-GPU setups.
warnings.filterwarnings('ignore', module='torch.nn.parallel')
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use")

def set_network_seed(cfg: dict):
    """
    Set random seeds for reproducibility.
    Ensures that random operations produce the same results across runs.

    Args:
        cfg (dict): A config dictionary containing the seed under the key 'SEED'.
    """
    random.seed(cfg["SEED"])                     # Set seed for Python's random module
    np.random.seed(cfg["SEED"])                  # Set seed for NumPy
    torch.manual_seed(cfg["SEED"])               # Set seed for PyTorch CPU
    torch.cuda.manual_seed(cfg["SEED"])          # Set seed for current GPU
    torch.cuda.manual_seed_all(cfg["SEED"])      # Set seed for all GPUs

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_snapshot(state, epoch, path):
    """
    Save the current model state and epoch to a file.

    Args:
        state: The model's state_dict.
        epoch: The current training epoch.
        path: File path to save the snapshot.
    """
    snapshot = {
        "MODEL_STATE": state,
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, path)
    print(f"Epoch {epoch} | Training snapshot saved at {path}")

def load_snapshot(model, path, rank):
    """
    Load a model snapshot and resume training from a saved epoch.

    Args:
        model: The PyTorch model to load the state into.
        path: Path to the saved snapshot file.
        rank: GPU device index for loading the snapshot correctly in multi-GPU setup.

    Returns:
        model: The model with loaded state.
        epoch: The epoch to resume training from.
    """
    loc = f"cuda:{rank}"                         # Ensure the model is loaded on the correct GPU
    snapshot = torch.load(path, map_location=loc)
    model.load_state_dict(snapshot["MODEL_STATE"])  # Load the saved weights into the model
    epoch = snapshot["EPOCHS_RUN"]               # Retrieve the saved epoch number
    print(f"Resuming training from snapshot at Epoch {epoch}")
    return model, epoch




def get_images(dataframe, modality):
    """
    Retrieves image file paths matching a specified modality from a DataFrame of patient paths.

    Args:
        dataframe (pd.DataFrame): DataFrame containing a 'Path' column with base directories.
        modality (str): Modality string to match in the filenames (e.g., "T1", "FLAIR", etc.).

    Returns:
        np.ndarray: Array of file paths that match the modality.
    """
    images = []

    # Iterate over each row in the DataFrame
    for index, row in dataframe.iterrows():
        # Build a search pattern to find images for the specified modality
        pattern = row['Path'] + "/" + "*" + modality + "*"

        # Find all file paths matching the pattern
        matched = glob.glob(pattern)

        # Add matched file paths to the images list
        images.extend(matched)

    # Convert the list to a NumPy array before returning
    images = np.array(images)
    return images


def append_or_create_excel(filename, trial, config, values):
    """
    Creates or appends configuration results to an Excel file.

    Args:
        filename (str): Path to the Excel file.
        trial (int): Identifier for the experiment trial.
        config (str): Configuration name.
        values (int): Values corresponding to each configuration.
    """
    # Ensure the directory exists; create it if it doesn't
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the Excel file already exists
    if os.path.isfile(filename):
        # Load existing Excel file into a DataFrame
        df = pd.read_excel(filename, index_col=0)
    else:
        # Create an empty DataFrame if file doesn't exist
        df = pd.DataFrame()

    # Convert single values to lists for consistent processing
    if isinstance(trial, (int, float)):
        trial = [trial]
    if isinstance(config, str):
        config = [config]
    if isinstance(values, (int, float)):
        values = [values]

    # Ensure trials are present in the DataFrame index
    if df.empty:
        # Add a dummy column to avoid errors when adding new rows
        df['Dummy'] = pd.NA

    for t in trial:
        if t not in df.index:
            # Initialize row with NA values for new trials
            df.loc[t] = [pd.NA] * len(df.columns)

    # Add or update values for the given configurations
    for conf, value in zip(config, values):
        df.loc[trial, conf] = value

    # Remove the dummy column if it's no longer needed
    if 'Dummy' in df.columns:
        df = df.drop(columns=['Dummy'])

    # Save the updated DataFrame back to the Excel file
    df.to_excel(filename)

    # Print status message
    print(f"Excel-Datei '{filename}' wurde {'aktualisiert' if os.path.isfile(filename) else 'erstellt'}.")


def medical_augment(level=1, prob=1):
    """
    Creates a medical image augmentation pipeline using randomized spatial and pixel-level transforms.

    Args:
        level (int): Intensity level for the augmentations (affects things like noise std).
        prob (float): Probability of applying the transformations.

    Returns:
        A Compose object that applies a randomized sequence of augmentations.
    """

    # Change prob_0 and prob_1 based on the number of transformations
    # Probability distribution used when selecting from pixel transforms (only one in this case)
    prob_0 = [1]

    # Equal probability distribution used for spatial transforms
    prob_1 = [1 / 3, 1 / 3, 1 / 3]

    # Pixel-level transformations (e.g., noise, contrast)
    pixel_transforms = [
        # Uncomment below to simulate low resolution if needed
        # RandSimulateLowResolution(prob=1 * prob, upsample_mode="trilinear", zoom_range=(0.9, 1)),

        # Add Gaussian noise with intensity controlled by `level` (currently disabled with prob=0)
        RandGaussianNoise(prob=1 * 0, mean=0.0, std=0.01 * level),

        # Uncomment below to adjust image contrast
        # RandAdjustContrast(prob=1 * prob, gamma=(0.8, 1.2), retain_stats=True)
    ]

    # Spatial transformations (e.g., rotation, flipping, affine translation)
    spatial_transforms = [
        RandRotate(range_x=0.15, range_y=0.15, range_z=0.15, prob=1 * prob, padding_mode="zeros"),
        RandAxisFlip(prob=1 * prob),

        # Uncomment below to enable scaling with affine transform
        # RandAffine(prob=1 * prob, scale_range=[0.1, 0.1, 0.1], padding_mode="zeros"),

        RandAffine(prob=1 * prob, translate_range=[5, 5, 2], padding_mode="zeros")
    ]

    # Combinations of 1 pixel and 2 spatial transformations in different orders
    transforms_1_2 = [
        Compose([
            OneOf(pixel_transforms, weights=prob_0),
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(spatial_transforms, weights=prob_1),
        ]),
        Compose([
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(pixel_transforms, weights=prob_0),
            OneOf(spatial_transforms, weights=prob_1),
        ]),
        Compose([
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(pixel_transforms, weights=prob_0),
        ])
    ]

    # Combination of 3 spatial transforms
    transforms_0_3 = [
        Compose([
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(spatial_transforms, weights=prob_1)
        ]),
    ]

    # Combination of 2 spatial transforms
    transforms_0_2 = [
        Compose([
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(spatial_transforms, weights=prob_1)
        ]),
    ]

    # Combinations of 1 pixel and 1 spatial transform in both possible orders
    transforms_1_1 = [
        Compose([
            OneOf(pixel_transforms, weights=prob_0),
            OneOf(spatial_transforms, weights=prob_1)
        ]),
        Compose([
            OneOf(spatial_transforms, weights=prob_1),
            OneOf(pixel_transforms, weights=prob_0)
        ])
    ]

    # Final augmentation pipeline: randomly choose one of the defined compositions above
    MedAugment = OneOf([
        OneOf(transforms_1_2, weights=[1 / 3, 1 / 3, 1 / 3]),  # 1 pixel + 2 spatial
        OneOf(transforms_0_3, weights=[1]),  # 3 spatial
        OneOf(transforms_0_2, weights=[1]),  # 2 spatial
        OneOf(transforms_1_1, weights=[1 / 2, 1 / 2])  # 1 pixel + 1 spatial
    ], weights=[1 / 4, 1 / 4, 1 / 4, 1 / 4])  # Equal chance to pick each group

    return MedAugment


def run(rank: int, world_size: int, cfg: dict) -> None:
    """
    Main training function to be launched on each GPU/device as part of distributed training.

    Args:
        rank (int): Local rank of the current process (used for assigning devices).
        world_size (int): Total number of processes (GPUs) participating in training.
        cfg (dict): Configuration dictionary with model, training, and data settings.
    """

    # Set seed for reproducibility (same random operations across devices)
    set_network_seed(cfg)

    # Setup the distributed process group (required for DDP training)
    setup(rank, world_size)

    # ----------------------------------------------------------------------------------
    # MODEL & OPTIMIZER SETUP
    # ----------------------------------------------------------------------------------

    # Initialize SegResNet or ResNet10 model for classification
    # In case of SuPreM
    model = SegResNet_Classification(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=1,
                                     out_channels=32, dropout_prob=0.0)
    # In case of Med3D
    # model = resnet10_new(num_classes=1)

    # Get the current working directory (like running 'pwd' in the terminal)
    current_path = os.getcwd()
    # Build the full path to the model checkpoint file
    # In case of SuPreM
    checkpoint_path = os.path.join(current_path, "Model", "supervised_suprem_segresnet_2100.pth")
    # In case of Med3D
    # checkpoint_path = os.path.join(current_path, "Model", "resnet_10_23dataset.pth")

    # Load the checkpoint into CPU memory
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract the model's state dictionary from the checkpoint
    if "state_dict" in checkpoint:
        model_dict = checkpoint["state_dict"]
    elif "net" in checkpoint:
        model_dict = checkpoint["net"]
    else:
        model_dict = checkpoint  # assume it's already a state_dict

    # Remove the 'module.' prefix from keys (used when model was trained with DataParallel)
    new_model_dict = {key.replace('module.', ''): value for key, value in model_dict.items()}

    # Ensure all weights are in float32 format for consistency
    for key in new_model_dict.keys():
        new_model_dict[key] = new_model_dict[key].float()

    # Load the weights into the model (non-strict in case some keys are missing or extra)
    model.load_state_dict(new_model_dict, strict=False)

    # Get the model's current state dictionary after loading
    model_state_dict = model.state_dict()

    # Compare weights to check if they exactly match the checkpoint
    mismatch_keys = []
    full = 0  # Total number of keys in the checkpoint
    part = 0  # Number of matching keys in the model

    for key in new_model_dict.keys():
        full += 1
        if key in model_state_dict:
            part += 1
            # Check if the weights are exactly equal
            if not torch.equal(model_state_dict[key], new_model_dict[key]):
                mismatch_keys.append(key)

    # Print a summary of mismatches
    print("\n=== Weight Comparison After Loading ===")
    if mismatch_keys:
        print(f"⚠️ The following {len(mismatch_keys)} layers are NOT identical to the checkpoint weights:")
        for key in mismatch_keys[:10]:  # Show only the first 10 mismatches
            print(f"  - {key}")
    else:
        print("✅ All model weights are IDENTICAL to the checkpoint!")

    # Final confirmation message
    if not mismatch_keys:
        print("\n🎉 Model weights perfectly match the checkpoint!")
    else:
        print("\n⚠️ Some weights differ from the checkpoint. Investigate further!")

    # Move the model to the specified device (e.g., GPU with index 'rank')
    model.to(rank)

    # If a checkpoint exists, resume training from that point
    model_name = cfg['MODEL_NAME']
    path = cfg["PATH_RESULT"] + "/MODELS/" + model_name + "_Config=" + str(cfg["CONFIG"]) + "_SPLIT=" + str(
        cfg["SPLIT"]) + "_latest" + '.pt'

    # If the path exists, resume from saved snapshot
    if os.path.exists(path):
        model, start_epoch = load_snapshot(model, path, rank)
    else:
        start_epoch = 1

    # Wrap the model for distributed training on GPU `rank`.
    # `find_unused_parameters=True` handles models with unused branches.
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer: AdamW with custom hyperparameters from config
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        eps=cfg['epsilon'],
        betas=cfg['betas'],
    )

    # Ensure folders exist for saving models and log files
    os.makedirs('FILES', exist_ok=True)
    os.makedirs('MODELS', exist_ok=True)

    # ----------------------------------------------------------------------------------
    # LOSS FUNCTION
    # ----------------------------------------------------------------------------------

    # Binary classification loss with logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS (MONAI)
    # ----------------------------------------------------------------------------------

    # Load dataset split info from Excel
    path_splits = cfg["path_splits"] + "_" + str(cfg["SPLIT"]) + ".xlsx"
    dataframe_train = pd.read_excel(path_splits, sheet_name='train')
    dataframe_validation = pd.read_excel(path_splits, sheet_name='validation')
    dataframe_test = pd.read_excel(path_splits, sheet_name='test')

    # Get list of image file paths for each split using the specified modality
    train_images = get_images(dataframe_train, cfg["modality"])
    validation_images = get_images(dataframe_validation, cfg["modality"])
    test_images = get_images(dataframe_test, cfg["modality"])

    # Convert labels from DataFrame to NumPy arrays
    train_labels = np.array(dataframe_train["Status"].tolist(), dtype=np.float64)
    validation_labels = np.array(dataframe_validation["Status"].tolist(), dtype=np.float64)
    test_labels = np.array(dataframe_test["Status"].tolist(), dtype=np.float64)

    # ----------------------------------------------------------------------------------
    # TRANSFORMATIONS & DATASET OBJECTS
    # ----------------------------------------------------------------------------------

    # Define data augmentation (applied only to training data)
    transformations = medical_augment(level=1, prob=0.5)
    transformations.set_random_state(seed=cfg["SEED"])

    # Compose data transformations (for MONAI-style datasets)
    train_transforms = Compose([EnsureChannelFirst(), transformations])
    validation_transforms = Compose([EnsureChannelFirst()])
    test_transforms = Compose([EnsureChannelFirst()])

    # Create MONAI ImageDataset objects with transformations applied
    train_dataset = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    validation_dataset = ImageDataset(image_files=validation_images, labels=validation_labels,
                                      transform=validation_transforms)
    test_dataset = ImageDataset(image_files=test_images, labels=test_labels, transform=test_transforms)

    # ----------------------------------------------------------------------------------
    # DISTRIBUTED DATALOADERS
    # ----------------------------------------------------------------------------------

    # Distributed sampler ensures each process gets a unique subset of data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg["SEED"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['WORKERS_PER_GPU'],
        sampler=train_sampler,
        persistent_workers=True,
        pin_memory=True,
    )

    # Validation loader with deterministic order (no shuffle)
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=5,
        num_workers=cfg['WORKERS_PER_GPU'],
        sampler=validation_sampler,
        persistent_workers=True,
        pin_memory=True,
    )

    # Test loader with deterministic order (no shuffle)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=5,
        num_workers=cfg['WORKERS_PER_GPU'],
        sampler=test_sampler,
        persistent_workers=True,
        pin_memory=True,
    )

    # ----------------------------------------------------------------------------------
    # START TRAINING
    # ----------------------------------------------------------------------------------

    train(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        epochs=cfg['EPOCHS'],
        rank=rank,
        cfg=cfg,
        start_epoch=start_epoch,
    )




def train(
        model: nn.Module,
        loss_func: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        rank: int,
        cfg: dict,
        start_epoch,
) -> Dict[str, List[float]]:
    """Execute training, validation, and test procedure with tracking and early stopping."""

    # Ensure reproducibility across runs
    set_network_seed(cfg)

    # ----------------------------------------------------------------------------------
    # METRICS SETUP
    # ----------------------------------------------------------------------------------
    number_classes = 1
    type_metric = 'binary'

    # Initialize torchmetrics objects for tracking loss, accuracy, AUC, and F1
    metric_train_loss = MeanMetric().to(rank)
    metric_validation_loss = MeanMetric().to(rank)
    metric_test_loss = MeanMetric().to(rank)
    metric_train_acc = BinaryAccuracy().to(rank)
    metric_validation_acc = Accuracy(task=type_metric, num_classes=number_classes).to(rank)
    metric_test_acc = Accuracy(task=type_metric, num_classes=number_classes).to(rank)
    metric_train_auc = BinaryAUROC().to(rank)
    metric_validation_auc = AUROC(task=type_metric, num_classes=number_classes).to(rank)
    metric_test_auc = AUROC(task=type_metric, num_classes=number_classes).to(rank)
    metric_train_f1 = BinaryF1Score().to(rank)
    metric_validation_f1 = F1Score(task=type_metric, num_classes=number_classes).to(rank)
    metric_test_f1 = F1Score(task=type_metric, num_classes=number_classes).to(rank)

    # Store metrics across epochs for plotting/logging
    metrics = {
        'train/loss': [],
        'validation/loss': [],
        'train/auc': [],
        'validation/auc': [],
    }

    # Track best validation performance for model saving
    best_validation = float("inf")

    # Initialize early stopping file with PATIENCE value
    early_stopping = open(cfg["PATH_RESULT"] + "/" + "CONFIG" + str(cfg["CONFIG"]) + "_SPLIT" + str(cfg["SPLIT"]) + "_early_stopping.txt", "w")
    early_stopping.write(str(cfg['PATIENCE']))
    early_stopping.close()

    # ----------------------------------------------------------------------------------
    # START TRAINING PROCEDURE
    # ----------------------------------------------------------------------------------
    for ep in range(start_epoch, epochs + 1):
        # Important for DistributedDataParallel (DDP):
        # Each process must manually set the current epoch in its sampler.
        # This ensures proper shuffling across epochs in a distributed setup.
        train_loader.sampler.set_epoch(ep)

        # ------------------------------------------------------------------------------
        # TRAINING LOOP (single epoch)
        # ------------------------------------------------------------------------------
        model.train()  # Set model to training mode (enables dropout, batchnorm updates, etc.)

        # Loop through each batch from the training DataLoader
        for x, y in tqdm(train_loader, leave=False):
            # Move input (x) and target (y) to the correct device (GPU)
            x = x.to(rank)
            y = y.to(rank)

            # Add a channel dimension to match model output shape: (B, 1)
            y = y.unsqueeze(1)

            # Forward pass: compute predicted outputs
            y_hat = model(x)

            # Compute the loss between predictions and ground truth
            batch_loss = loss_func(y_hat, y)

            # Backpropagation and optimizer step
            optimizer.zero_grad()  # Clear existing gradients
            batch_loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights

            # Update loss metric tracker
            metric_train_loss(batch_loss)

            # Prepare ground truth and predictions for metric computation
            y_metric = y.float().squeeze(1)
            y_hat_metric = torch.sigmoid(y_hat).squeeze(1)  # Apply sigmoid to logits
            y_hat_metric = torch.tensor(y_hat_metric).to(rank)  # Ensure it's on the correct device

            # Update training metrics
            metric_train_acc(y_hat_metric, y_metric)
            metric_train_auc(y_hat_metric, y_metric)
            metric_train_f1(y_hat_metric, y_metric)

        # ------------------------------------------------------------------------------
        # VALIDATION LOOP
        # ------------------------------------------------------------------------------

        model.eval()  # Set model to evaluation mode (disables dropout, uses running stats for batchnorm)

        # Loop through validation dataset
        for x, y in tqdm(validation_loader, leave=False):
            x = x.to(rank)
            y = y.to(rank)

            # Disable gradient tracking for inference (saves memory and speed)
            with torch.no_grad():
                y = y.unsqueeze(1)  # Add channel dimension
                y_hat = model(x)  # Forward pass
                batch_loss = loss_func(y_hat, y)  # Compute loss

            # Update validation loss
            metric_validation_loss(batch_loss)

            # Prepare predictions and labels for metric calculation
            y_metric = y.float().squeeze(1)  # Ground truth: shape (B,)
            y_hat_metric = torch.sigmoid(y_hat).squeeze(1)  # Apply sigmoid to logits: shape (B,)
            y_hat_metric = torch.tensor(y_hat_metric).to(rank)

            # Update validation metrics
            metric_validation_acc(y_hat_metric, y_metric)
            metric_validation_auc(y_hat_metric, y_metric)
            metric_validation_f1(y_hat_metric, y_metric)

        # ------------------------------------------------------------------------------
        # TEST LOOP (run every epoch to monitor test performance)
        # ------------------------------------------------------------------------------

        for x, y in tqdm(test_loader, leave=False):
            x = x.to(rank)
            y = y.to(rank)

            with torch.no_grad():
                y = y.unsqueeze(1)
                y_hat = model(x)
                batch_loss = loss_func(y_hat, y)

            # Update test loss
            metric_test_loss(batch_loss)

            # Prepare predictions and labels for metric calculation
            y_metric = y.float().squeeze(1)
            y_hat_metric = torch.sigmoid(y_hat).squeeze(1)
            y_hat_metric = torch.tensor(y_hat_metric).to(rank)

            # Update test metrics
            metric_test_acc(y_hat_metric, y_metric)
            metric_test_auc(y_hat_metric, y_metric)
            metric_test_f1(y_hat_metric, y_metric)

        # ------------------------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------------------------

        # Compute final averaged metrics for this epoch.
        # torchmetrics automatically synchronizes across GPUs in DDP mode.
        # `compute()` returns the aggregated metric over all updates.
        ep_train_loss = float(metric_train_loss.compute())
        ep_validation_loss = float(metric_validation_loss.compute())
        ep_test_loss = float(metric_test_loss.compute())

        ep_train_acc = float(metric_train_acc.compute())
        ep_validation_acc = float(metric_validation_acc.compute())
        ep_test_acc = float(metric_test_acc.compute())

        ep_train_auc = float(metric_train_auc.compute())
        ep_validation_auc = float(metric_validation_auc.compute())
        ep_test_auc = float(metric_test_auc.compute())

        ep_train_f1 = float(metric_train_f1.compute())
        ep_validation_f1 = float(metric_validation_f1.compute())
        ep_test_f1 = float(metric_test_f1.compute())

        # Save validation, training, and test performance metrics for the current epoch to an Excel file

        # Construct the output filename for performance tracking
        filename = cfg["PATH_RESULT"] + '/FILES/Performance' "_CONFIG_" + str(cfg["CONFIG"]) + ".xlsx"

        # Define the metric names that will be saved
        saved_metrics = ["LOSS", "ACC", "AUC", "F1"]

        # Create a unique identifier string based on config and trial number
        configs = "Config=" + str(cfg["CONFIG"]) + "_Trial=" + str(cfg["SPLIT"])

        # Build full metric names for validation, training, and test sets using the identifiers
        identification_val = ["VAL_" + item + "_" + configs for item in saved_metrics]
        identification_train = ["TRAIN_" + item + "_" + configs for item in saved_metrics]
        identification_test = ["TEST_" + item + "_" + configs for item in saved_metrics]

        # Combine all identifiers into a single list
        identification = identification_val + identification_train + identification_test

        # Store current epoch number in a list (required for Excel logging)
        epoch = [ep]

        # Print metric identifiers for debugging/logging purposes
        print(identification)

        # ----------------------------------------------------------------------------------
        # RESET METRIC TRACKERS FOR NEXT EPOCH
        # ----------------------------------------------------------------------------------

        # Reset all torchmetrics objects so they don't carry over into the next epoch
        metric_train_loss.reset()
        metric_validation_loss.reset()
        metric_test_loss.reset()

        metric_train_acc.reset()
        metric_validation_acc.reset()
        metric_test_acc.reset()

        metric_train_auc.reset()
        metric_validation_auc.reset()
        metric_test_auc.reset()

        metric_train_f1.reset()
        metric_validation_f1.reset()
        metric_test_f1.reset()

        # ----------------------------------------------------------------------------------
        # APPEND CURRENT EPOCH METRICS TO IN-MEMORY DICTIONARY
        # ----------------------------------------------------------------------------------

        # These are tracked in memory
        metrics['train/loss'].append(ep_train_loss)
        metrics['validation/loss'].append(ep_validation_loss)
        metrics['train/auc'].append(ep_train_auc)
        metrics['validation/auc'].append(ep_validation_auc)

        # ----------------------------------------------------------------------------------
        # MASTER NODE ONLY (rank 0)
        # ----------------------------------------------------------------------------------
        # Only the process with rank 0 will perform actions like logging, saving models, and checking early stopping
        if rank == 0:
            # Store all computed metrics from this epoch into a list
            values = [[
                ep_validation_loss, ep_validation_acc, ep_validation_auc, ep_validation_f1,
                ep_train_loss, ep_train_acc, ep_train_auc, ep_train_f1,
                ep_test_loss, ep_test_acc, ep_test_auc, ep_test_f1
            ]]

            # Write current epoch metrics to the main performance tracking Excel file
            append_or_create_excel(filename, identification, epoch, values)

            # ------------------------------------------------------------------------------
            # EARLY STOPPING LOGIC
            # ------------------------------------------------------------------------------

            # Read the current early stopping counter from file
            early_stopping_path = f"{cfg['PATH_RESULT']}/CONFIG{cfg['CONFIG']}_SPLIT{cfg['SPLIT']}_early_stopping.txt"
            with open(early_stopping_path, "r") as f:
                early_stopping_counter = int(f.read())
            print(early_stopping_counter)

            # Decrease patience counter by 1 and update the file
            with open(early_stopping_path, "w") as f:
                f.write(str(early_stopping_counter - 1))

            # Print epoch summary to console
            print(
                'EP: {:3} | LOSS: T {:.3f} V {:.3f} | AUC: T {:.3f} V {:.3f}'.format(
                    ep, ep_train_loss, ep_validation_loss, ep_train_auc, ep_validation_auc
                )
            )

            # ------------------------------------------------------------------------------
            # SAVE LATEST MODEL SNAPSHOT (always)
            # ------------------------------------------------------------------------------
            model_name = f"{cfg['MODEL_NAME']}_Config={cfg['CONFIG']}_SPLIT={cfg['SPLIT']}_latest.pt"
            save_snapshot(model.module.state_dict(), ep, os.path.join('MODELS', model_name))

            # ------------------------------------------------------------------------------
            # CHECK FOR BEST VALIDATION PERFORMANCE
            # ------------------------------------------------------------------------------

            # Path to the best performance tracking Excel
            file_path = f"{cfg['PATH_RESULT']}/FILES/Best_Performance_CONFIG_{cfg['CONFIG']}.xlsx"
            val_metric_key = f"VAL_Metric=LOSS_Config={cfg['CONFIG']}"

            # If the file exists, try to retrieve the best validation score so far
            if os.path.exists(file_path):
                df = pd.read_excel(file_path, index_col=0)
                if val_metric_key in df.columns and cfg["SPLIT"] in df.index:
                    stored_value = df.loc[cfg["SPLIT"], val_metric_key]
                    best_validation = stored_value
                    output_message = f"Stored Validation Loss for trial {cfg['SPLIT']}: {stored_value}"
                else:
                    best_validation = float("inf")
                    output_message = f"Key '{val_metric_key}' or trial '{cfg['SPLIT']}' not found in the file."
            else:
                best_validation = float("inf")
                output_message = "File does not exist."

            # ------------------------------------------------------------------------------
            # SAVE BEST MODEL (only if validation improves)
            # ------------------------------------------------------------------------------
            if ep_validation_loss < best_validation:
                best_validation = ep_validation_loss

                # Save best model with a different name (without _latest)
                model_name = f"{cfg['MODEL_NAME']}_Config={cfg['CONFIG']}_SPLIT={cfg['SPLIT']}.pt"
                save_snapshot(model.module.state_dict(), ep, os.path.join('MODELS', model_name))

                # Log best performance details
                filename = f"{cfg['PATH_RESULT']}/FILES/Best_Performance_CONFIG_{cfg['CONFIG']}.xlsx"
                trial = [cfg["SPLIT"]]
                identification = [
                    f"VAL_Metric=LOSS_Config={cfg['CONFIG']}",
                    f"TEST_Metric=AUC_Config={cfg['CONFIG']}",
                    f"Epoch_Config={cfg['CONFIG']}"
                ]
                values = [[best_validation], [ep_test_auc], [ep]]
                append_or_create_excel(filename, trial, identification, values)

                # Reset early stopping counter back to original patience value
                with open(early_stopping_path, "w") as f:
                    f.write(str(cfg['PATIENCE']))

            # ------------------------------------------------------------------------------
            # EARLY STOPPING CHECK
            # ------------------------------------------------------------------------------
            with open(early_stopping_path, "r") as f:
                early_stopping_counter = int(f.read())

            # If patience is exhausted and model shows decent learning, stop training early
            if early_stopping_counter == -1 and ep_validation_auc >= 0.7 and ep_train_auc >= 0.7:
                break

        # ------------------------------------------------------------------------------
        # OPTIONAL HARD STOP IF MODEL IS OVERFITTING (very high AUC)
        # ------------------------------------------------------------------------------
        if ep_train_auc >= 0.99:
            break

    # Return the tracked metrics dictionary (useful for visualization or debugging)
    return metrics


def read_yml(filepath: str) -> dict:
    """
    Load a YAML configuration file into memory as a Python dictionary.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


def setup(rank: int, world_size: int) -> None:
    """
    Initialize the process group for Distributed Data Parallel (DDP) training.

    Args:
        rank (int): Rank of the current process (used as device ID).
        world_size (int): Total number of processes to spawn (e.g., number of GPUs).
    """
    # Set master address to localhost for single-node training
    os.environ['MASTER_ADDR'] = 'localhost'

    # Try to dynamically determine a free master port (using SLURM job ID)
    try:
        default_port = os.environ['SLURM_JOB_ID'][-4:]
        default_port = int(default_port) + 15000  # Ensure it's in a safe range
    except Exception:
        default_port = 12910  # Fallback to a hardcoded port

    os.environ['MASTER_PORT'] = str(default_port)

    # Initialize the PyTorch distributed backend
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f'INITIALIZED RANK {rank}')



def cleanup():
    """
    Properly terminate the DDP process group.
    Should be called at the end of training to free resources.
    """
    dist.destroy_process_group()


def main() -> None:
    """
    Main execution function for training setup and execution.
    Handles config loading, DDP spawning, and training orchestration.
    """

    # If necessary: Define which dataset split to use — could iterate over multiple if needed
    for SPLIT in [0]:

        # ------------------------------------------------------------------------------
        # Parse command-line arguments
        # ------------------------------------------------------------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-c', '--config',
            default='/home/hagen/Downloads/temp/config_2.yml',
            type=str,
            help="Path to YAML config file"
        )
        parser.add_argument(
            "--config_id",
            type=int,
            required=False,
            help="Optional configuration ID (used as split index)"
        )
        args = parser.parse_args()

        print(args.config)

        # ------------------------------------------------------------------------------
        # Load configuration and update with runtime parameters
        # ------------------------------------------------------------------------------
        config_id = args.config_id
        cfg = read_yml(args.config)        # Load YAML file as dictionary
        set_network_seed(cfg)              # Set global seed for reproducibility
        cfg["SPLIT"] = config_id           # Assign split ID to config for tracking

        print(cfg)
        print(f"CONFIG ID: {config_id}")

        # Change working directory to result path (useful for file saving paths)
        os.chdir(cfg["PATH_RESULT"])

        # ------------------------------------------------------------------------------
        # Distributed Training Setup
        # ------------------------------------------------------------------------------
        world_size = torch.cuda.device_count()  # Get number of available GPUs

        # Spawn a separate process for each GPU. Each process will call `run(rank, world_size, cfg)`
        # `rank` is automatically assigned (0 to world_size - 1)
        mp.spawn(
            run,                     # Function to run in each process
            args=(world_size, cfg),  # Shared args to pass to each process
            nprocs=world_size        # Number of processes (GPUs)
        )


# Entry point: ensures `main()` runs only when script is executed directly
if __name__ == '__main__':
    main()

# Notes for developers:
# - Early stopping logic is implemented based on a counter stored in a .txt file.
# - Data loading is based on Excel splits and MONAI’s `ImageDataset`.
# - Metrics include: Loss, Accuracy, AUC, and F1 for train/val/test using torchmetrics.