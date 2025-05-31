import argparse
import os
import warnings
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Accuracy, MeanMetric, AUROC, F1Score
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAccuracy
from tqdm import tqdm
import monai
import pandas as pd
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, RandSimulateLowResolution, RandGaussianNoise, RandAdjustContrast, RandRotate, RandAxisFlip, RandAffine, OneOf, EnsureChannelFirst, Resize
import random
import glob
import torch.nn as nn
import torch.nn.functional as F
from Helper.config import models_genesis_config
from Helper.utils import *
import gc

# Configure PyTorch's CUDA memory allocator to limit the maximum chunk size for memory splits.
# Helps reduce fragmentation on certain workloads.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# Suppress legacy warnings related to torch.nn.parallel
warnings.filterwarnings('ignore', module='torch.nn.parallel')
.
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use")


# Function to set all relevant random seeds for reproducibility across runs.
def set_network_seed(cfg: dict):
    random.seed(cfg["SEED"])                # Python random seed
    np.random.seed(cfg["SEED"])             # NumPy seed

    torch.manual_seed(cfg["SEED"])          # CPU seed for PyTorch
    torch.cuda.manual_seed(cfg["SEED"])     # CUDA seed for current device
    torch.cuda.manual_seed_all(cfg["SEED"]) # CUDA seed for all devices (if multiple GPUs)

    # Ensure deterministic behavior (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Redundant, but ensures NumPy seed is set again (not harmful)
    np.random.seed(cfg["SEED"])


# Save a training checkpoint including model state, epoch number, and optimizer state.
def save_snapshot(state, epoch, path, optimizer):
    snapshot = {
        "MODEL_STATE": state,
        "EPOCHS_RUN": epoch,
        "OPTIMIZER_STATE_DICT": optimizer.state_dict()
    }
    torch.save(snapshot, path)
    print(f"Epoch {epoch} | Training snapshot saved at {path}")


# Load a training checkpoint from disk and restore model and optimizer states.
def load_snapshot(model, path, rank, optimizer):
    loc = f"cuda:{rank}"                            # Ensure loading happens on the correct GPU
    snapshot = torch.load(path, map_location=loc)   # Load the checkpoint file

    model.load_state_dict(snapshot["MODEL_STATE"])  # Restore model parameters
    epoch = snapshot["EPOCHS_RUN"]                  # Restore epoch count
    optimizer.load_state_dict(snapshot['OPTIMIZER_STATE_DICT'])  # Restore optimizer state

    print(f"Resuming training from snapshot at Epoch {epoch}")
    return model, epoch, optimizer                   # Return the restored model and state








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
        #8
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

        # Klassifikationskopf
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
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
        classification_output = self.fc(pooled_features)
        # output classification and reconstruction head
        return self.out, classification_output




def get_images(dataframe, modality):
    images = []

    # Print the "Patients" column to verify which patients are processed
    print(dataframe["Patients"])

    # Loop through each row in the dataframe (one per patient or sample)
    for index, row in dataframe.iterrows():
        # Construct a glob search pattern to find files for the specified modality
        pattern = row['Path'] + "/" + "*" + modality + "*"

        # Find all files matching the pattern (e.g., *.nii.gz for a modality)
        matched = glob.glob(pattern)

        # Add the matched files to the image list
        images.extend(matched)

    # Convert the list to a NumPy array for compatibility with later code
    images = np.array(images)
    return images


def append_or_create_excel(filename, trial, config, values):
    # Ensure that the output directory exists, create it if necessary
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Load existing Excel file if available, otherwise start with an empty DataFrame
    if os.path.isfile(filename):
        df = pd.read_excel(filename, index_col=0)
    else:
        df = pd.DataFrame()

    # Ensure all input values are lists (for consistent handling)
    if isinstance(trial, (int, float)):
        trial = [trial]
    if isinstance(config, str):
        config = [config]
    if isinstance(values, (int, float)):
        values = [values]

    # If the DataFrame is empty, add a dummy column to enable row assignment
    if df.empty:
        df['Dummy'] = pd.NA

    # Ensure that each trial exists as an index in the DataFrame
    for t in trial:
        if t not in df.index:
            df.loc[t] = [pd.NA] * len(df.columns)

    # Update the DataFrame with new values for each config
    for conf, value in zip(config, values):
        df.loc[trial, conf] = value

    # Remove dummy column if it is no longer needed
    if 'Dummy' in df.columns:
        df = df.drop(columns=['Dummy'])

    # Save the DataFrame back to the Excel file
    df.to_excel(filename)

    print(f"Excel file '{filename}' has been {'updated' if os.path.isfile(filename) else 'created'}.")

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


def run(rank: int, world_size: int, cfg: dict, conf) -> None:
    # Set all random seeds for reproducibility across runs
    set_network_seed(cfg)

    """Kick off training."""
    # Initialize the distributed training environment for this process (based on rank)
    setup(rank, world_size)

    # ----------------------------------------------------------------------------------
    # MODEL & OPTIMIZER
    # ----------------------------------------------------------------------------------

    # Create the model and move it to the GPU corresponding to the current rank
    model = UNet3DWithClassification().to(rank)

    # Define the checkpoint path for resuming training
    model_name = cfg['MODEL_NAME']
    path = (
        cfg["PATH_RESULT"] + "/MODELS/" + model_name +
        f"_Config={cfg['CONFIG']}_SPLIT={cfg['SPLIT']}_latest.pt"
    )

    # Create the optimizer (AdamW in this case)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        eps=cfg['epsilon'],
        betas=cfg['betas'],
    )

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(conf.patience * 0.8),  # Adjust step size based on patience
        gamma=0.5  # Reduce LR by factor 0.5
    )

    # If a previous checkpoint exists, resume training from that snapshot
    if os.path.exists(path):
        model, start_epoch, optimizer = load_snapshot(model, path, rank, optimizer)
    else:
        start_epoch = 1  # Otherwise, start from scratch

    # Wrap the model for Distributed Data Parallel (DDP) training
    model = DDP(model, device_ids=[rank])

    # Create directories for saving model checkpoints and results
    os.makedirs('FILES', exist_ok=True)
    os.makedirs('MODELS', exist_ok=True)

    # ----------------------------------------------------------------------------------
    # LOSS FUNCTIONS
    # ----------------------------------------------------------------------------------

    # Segmentation loss (custom weighted loss for sparse/ROI data)
    loss_func = WeightedLossWithNonzeroRatio_ROI()

    # Binary classification loss for the classification head
    loss_func_bin = torch.nn.BCEWithLogitsLoss()

    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS (MONAI)
    # ----------------------------------------------------------------------------------

    # Build the full path to the Excel file that defines the data splits
    path_splits = cfg["path_splits"]
    path_splits = path_splits + "_" + str(cfg["SPLIT"]) + ".xlsx"

    # Load the data split Excel file into three DataFrames: train, validation, test
    dataframe_train = pd.read_excel(path_splits, sheet_name='train')
    dataframe_validation = pd.read_excel(path_splits, sheet_name='validation')
    dataframe_test = pd.read_excel(path_splits, sheet_name='test')

    # Extract paths to modality-specific images for each split using the helper function
    train_images = get_images(dataframe_train, cfg["modality"])
    validation_images = get_images(dataframe_validation, cfg["modality"])
    test_images = get_images(dataframe_test, cfg["modality"])

    # Extract binary classification labels (e.g., "Status") and convert to NumPy float arrays
    train_labels = dataframe_train["Status"].tolist()
    train_labels = np.array(train_labels, dtype=np.float64)

    validation_labels = dataframe_validation["Status"].tolist()
    validation_labels = np.array(validation_labels, dtype=np.float64)

    test_labels = dataframe_test["Status"].tolist()
    test_labels = np.array(test_labels, dtype=np.float64)

    # Create the transformation pipeline with data augmentation
    transformations = medical_augment(level=1, prob=0.5)

    # Set the random seed for the transform pipeline to ensure reproducibility
    transformations.set_random_state(seed=cfg["SEED"])

    # Define the preprocessing and augmentation for training and evaluation
    # All images are resized to 80x80x80, and training includes augmentations
    train_transforms = Compose([
        EnsureChannelFirst(),  # Ensures input shape is (C, D, H, W)
        Resize((80, 80, 80)),  # Resize 3D volumes to fixed shape
        transformations  # Apply randomized medical augmentations
    ])

    # Validation and test transforms only include resizing and channel formatting (no augmentation)
    validation_transforms = Compose([
        EnsureChannelFirst(),
        Resize((80, 80, 80))
    ])

    test_transforms = Compose([
        EnsureChannelFirst(),
        Resize((80, 80, 80))
    ])

    # Create datasets using MONAI's ImageDataset
    train_dataset = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    validation_dataset = ImageDataset(image_files=validation_images, labels=validation_labels,
                                      transform=validation_transforms)
    test_dataset = ImageDataset(image_files=test_images, labels=test_labels, transform=test_transforms)

    # ----------------------------------------------------------------------------------
    # DISTRIBUTED SAMPLERS & DATALOADERS
    # ----------------------------------------------------------------------------------

    # Create a distributed sampler for the training dataset
    # Ensures that each GPU receives a unique subset of the data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,  # Total number of GPUs
        rank=rank,  # Index of the current GPU
        shuffle=True,  # Enable shuffling for training
        seed=cfg["SEED"]
    )

    # Define the DataLoader for training (parallel data loading and pinned memory enabled)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['WORKERS_PER_GPU'],
        sampler=train_sampler,
        persistent_workers=True,
        pin_memory=True,
    )

    # Create a distributed sampler for validation (no shuffling!)
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Validation DataLoader (batch size 1 to evaluate one sample at a time)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        num_workers=cfg['WORKERS_PER_GPU'],
        sampler=validation_sampler,
        persistent_workers=True,
        pin_memory=True,
    )

    # Distributed sampler for test data (same structure as validation)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Test DataLoader (batch size 1, no shuffling)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg['WORKERS_PER_GPU'],
        sampler=test_sampler,
        persistent_workers=True,
        pin_memory=True,
    )

    # ----------------------------------------------------------------------------------
    # CONDUCT TRAINING
    # ----------------------------------------------------------------------------------

    # Start the training process by calling the `train` function.
    train(
        model=model,
        loss_func=loss_func,  # Custom loss for segmentation output
        loss_func_bin=loss_func_bin,  # BCE loss for binary classification head
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        epochs=cfg['EPOCHS'],  # Total number of training epochs
        rank=rank,  # Rank of the current process (GPU)
        cfg=cfg,  # Configuration dictionary
        start_epoch=start_epoch,  # Resume from this epoch (in case of checkpointing)
        scheduler=scheduler,  # Learning rate scheduler
        conf=conf
    )


def train(
        model: nn.Module,
        loss_func: nn.Module,
        loss_func_bin: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        rank: int,
        cfg: dict,
        start_epoch,
        scheduler,
        conf,
) -> Dict[str, List[float]]:
    """Execute training procedure."""

    # Ensure deterministic behavior (re-set seed in case of forked workers etc.)
    set_network_seed(cfg)

    # ----------------------------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------------------------

    number_classes = 1  # Binary classification task
    type_metric = 'binary'  # Type of classification metrics

    # Define training, validation, and test loss trackers
    metric_train_loss = MeanMetric().to(rank)
    metric_validation_loss = MeanMetric().to(rank)
    metric_test_loss = MeanMetric().to(rank)

    # Accuracy metrics for each phase
    metric_train_acc = BinaryAccuracy().to(rank)
    metric_validation_acc = Accuracy(task=type_metric, num_classes=number_classes).to(rank)
    metric_test_acc = Accuracy(task=type_metric, num_classes=number_classes).to(rank)

    # AUROC metrics for each phase
    metric_train_auc = BinaryAUROC().to(rank)
    metric_validation_auc = AUROC(task=type_metric, num_classes=number_classes).to(rank)
    metric_test_auc = AUROC(task=type_metric, num_classes=number_classes).to(rank)

    # F1 score metrics for each phase
    metric_train_f1 = BinaryF1Score().to(rank)
    metric_validation_f1 = F1Score(task=type_metric, num_classes=number_classes).to(rank)
    metric_test_f1 = F1Score(task=type_metric, num_classes=number_classes).to(rank)

    # ------------------------------------------------------------------------------
    # METRIC TRACKING INITIALIZATION
    # ------------------------------------------------------------------------------

    # Dictionary to track training and validation metrics over all epochs
    metrics = {
        'train/loss': [],
        'validation/loss': [],
        'train/auc': [],
        'validation/auc': [],
    }

    # Initialize best validation loss with infinity to enable saving the best model later
    best_validation = float("inf")

    # Create/reset early stopping counter file for this config/split
    early_stopping_path = os.path.join(
        cfg["PATH_RESULT"],
        f"CONFIG{cfg['CONFIG']}_SPLIT{cfg['SPLIT']}_early_stopping.txt"
    )
    with open(early_stopping_path, "w") as early_stopping:
        early_stopping.write(str(cfg['PATIENCE']))

    # ------------------------------------------------------------------------------
    # START TRAINING LOOP
    # ------------------------------------------------------------------------------

    for ep in range(start_epoch, epochs + 1):
        # Synchronize all processes before each epoch
        dist.barrier()

        # Advance the learning rate scheduler, if configured
        if cfg['scheduler']:
            scheduler.step(ep)

        # IMPORTANT: Set epoch for DistributedSampler to reshuffle data consistently across workers
        train_loader.sampler.set_epoch(ep)

        # ------------------------------------------------------------------------------
        # TRAINING PHASE
        # ------------------------------------------------------------------------------

        model.train()  # Set model to training mode

        # Iterate over all batches in the training set
        for x, y in tqdm(train_loader, leave=False):
            # Clear CUDA memory (defensive memory handling for large 3D data)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            # Prepare input and label tensors
            x = x.numpy()
            y = y.to(rank).unsqueeze(1)
            in_x, out_x = generate_pair(x, conf)  # Generate a pair of input/output volumes
            in_x = torch.from_numpy(in_x).float().to(rank)
            out_x = torch.from_numpy(out_x).float().to(rank)

            # Forward pass through the model
            y_hat, y_class = model(in_x)

            # Compute segmentation and classification losses
            batch_loss = loss_func(y_hat, out_x)
            batch_loss_bin = loss_func_bin(y_class, y)

            # Combine losses (classification loss currently disabled via weight 0)
            total_loss = batch_loss * 1 + batch_loss_bin * 0

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update training loss metric
            metric_train_loss(total_loss)

            # Prepare predictions and targets for classification metrics
            y_metric = y.float().squeeze(1)
            y_hat_metric = torch.sigmoid(y_class).squeeze(1).to(rank)

            # Update training metrics
            metric_train_acc(y_hat_metric, y_metric)
            metric_train_auc(y_hat_metric, y_metric)
            metric_train_f1(y_hat_metric, y_metric)

        # ------------------------------------------------------------------------------
        # VALIDATION PHASE
        # ------------------------------------------------------------------------------

        # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
        model.eval()

        # Loop through the validation dataset to compute loss and metrics
        for x, y in tqdm(validation_loader, leave=False):
            # Prepare the batch
            x = x.numpy()
            y = y.to(rank).unsqueeze(1)  # Ensure label has shape (B, 1)

            # Generate paired input/output patches
            in_x, out_x = generate_pair(x, conf)
            in_x = torch.from_numpy(in_x).float().to(rank)
            out_x = torch.from_numpy(out_x).float().to(rank)

            # Inference without gradient tracking
            with torch.no_grad():
                y_hat, y_class = model(in_x)  # Forward pass

                # Compute segmentation and classification losses
                batch_loss = loss_func(y_hat, out_x)
                batch_loss_bin = loss_func_bin(y_class, y)
                total_loss = batch_loss * 1 + batch_loss_bin * 0  # Only segmentation loss used here

            # Update validation loss tracker
            metric_validation_loss(total_loss)

            # Prepare tensors for metric calculation
            y_metric = y.float().squeeze(1)
            y_hat_metric = torch.sigmoid(y_class).squeeze(1).to(rank)

            # Update validation classification metrics
            metric_validation_acc(y_hat_metric, y_metric)
            metric_validation_auc(y_hat_metric, y_metric)
            metric_validation_f1(y_hat_metric, y_metric)

        # ------------------------------------------------------------------------------
        # TEST PHASE
        # ------------------------------------------------------------------------------

        # Loop through the test dataset in the same way as validation
        for x, y in tqdm(test_loader, leave=False):
            # Prepare batch
            x = x.numpy()
            y = y.to(rank).unsqueeze(1)

            # Generate paired input/output data
            in_x, out_x = generate_pair(x, conf)
            in_x = torch.from_numpy(in_x).float().to(rank)
            out_x = torch.from_numpy(out_x).float().to(rank)

            # Forward pass without gradient computation
            with torch.no_grad():
                # NOTE: `out_x` is mistakenly used as input here — verify if this is intentional or a bug
                y_hat, y_class = model(out_x)

                batch_loss = loss_func(y_hat, out_x)
                batch_loss_bin = loss_func_bin(y_class, y)
                total_loss = batch_loss * 1 + batch_loss_bin * 0

            # Update test loss metric
            metric_test_loss(total_loss)

            # Compute test metrics
            y_metric = y.float().squeeze(1)
            y_hat_metric = torch.sigmoid(y_class).squeeze(1).to(rank)

            metric_test_acc(y_hat_metric, y_metric)
            metric_test_auc(y_hat_metric, y_metric)
            metric_test_f1(y_hat_metric, y_metric)

        # ------------------------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------------------------

        # Compute final aggregated metrics for this epoch
        # Note: `torchmetrics` handles synchronization across all distributed processes (DDP-compatible)

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

        # ------------------------------------------------------------------------------
        # SAVE METRICS TO EXCEL
        # ------------------------------------------------------------------------------

        # Construct filename for performance tracking across epochs
        filename = os.path.join(cfg["PATH_RESULT"], f'FILES/Performance_CONFIG_{cfg["CONFIG"]}.xlsx')

        # Define which metrics to save and construct unique column headers per split
        saved_metrics = ["LOSS", "ACC", "AUC", "F1"]
        config_id = f"Config={cfg['CONFIG']}_Trial={cfg['SPLIT']}"

        identification_val = [f"VAL_{metric}_{config_id}" for metric in saved_metrics]
        identification_train = [f"TRAIN_{metric}_{config_id}" for metric in saved_metrics]
        identification_test = [f"TEST_{metric}_{config_id}" for metric in saved_metrics]

        identification = identification_val + identification_train + identification_test
        epoch = [ep]

        # Print metric keys being written (debug/logging)
        print(identification)

        # ------------------------------------------------------------------------------
        # RESET METRIC STATES FOR NEXT EPOCH
        # ------------------------------------------------------------------------------

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

        # ------------------------------------------------------------------------------
        # UPDATE METRIC HISTORY FOR TRACKING
        # ------------------------------------------------------------------------------

        # Append current epoch metrics to history tracker
        metrics['train/loss'].append(ep_train_loss)
        metrics['validation/loss'].append(ep_validation_loss)
        metrics['train/auc'].append(ep_train_auc)
        metrics['validation/auc'].append(ep_validation_auc)

        # ------------------------------------------------------------------------------
        # MASTER NODE OPERATIONS (only executed by rank 0)
        # ------------------------------------------------------------------------------

        # In a DDP setup, we designate rank 0 as the master node
        # Only rank 0 handles logging, printing, early stopping logic, and model saving
        if rank == 0:
            # Prepare performance metrics from this epoch for logging
            values = [[
                ep_validation_loss, ep_validation_acc, ep_validation_auc, ep_validation_f1,
                ep_train_loss, ep_train_acc, ep_train_auc, ep_train_f1,
                ep_test_loss, ep_test_acc, ep_test_auc, ep_test_f1
            ]]

            # Save metrics to the tracking Excel file
            append_or_create_excel(filename, identification, epoch, values)

            # Load current early stopping counter from file
            early_stopping_path = os.path.join(
                cfg["PATH_RESULT"],
                f"CONFIG{cfg['CONFIG']}_SPLIT{cfg['SPLIT']}_early_stopping.txt"
            )
            with open(early_stopping_path, "r") as f:
                early_stopping_counter = int(f.read())

            print(early_stopping_counter)  # For debugging

            # Decrease patience by 1 (performance not improved yet)
            with open(early_stopping_path, "w") as f:
                f.write(str(early_stopping_counter - 1))

            # Print current progress
            print(
                f'EP: {ep:3} | LOSS: T {ep_train_loss:.3f} V {ep_validation_loss:.3f} | '
                f'AUC: T {ep_train_auc:.3f} V {ep_validation_auc:.3f}'
            )

            # Save latest model checkpoint for recovery/resuming
            latest_model_name = f"{cfg['MODEL_NAME']}_Config={cfg['CONFIG']}_SPLIT={cfg['SPLIT']}_latest.pt"
            save_snapshot(model.module.state_dict(), ep, os.path.join('MODELS', latest_model_name), optimizer)

            # Save best model if validation loss improves
            if ep_validation_loss < best_validation:
                best_validation = ep_validation_loss

                best_model_name = f"{cfg['MODEL_NAME']}_Config={cfg['CONFIG']}_SPLIT={cfg['SPLIT']}.pt"
                save_snapshot(model.module.state_dict(), ep, os.path.join('MODELS', best_model_name), optimizer)

                # Log best metrics to separate file
                best_perf_filename = os.path.join(cfg["PATH_RESULT"],
                                                  f'FILES/Best_Performance_CONFIG_{cfg["CONFIG"]}.xlsx')
                trial = [cfg["SPLIT"]]
                identification = [
                    f"VAL_Metric=LOSS_Config={cfg['CONFIG']}",
                    f"TEST_Metric=AUC_Config={cfg['CONFIG']}",
                    f"Epoch_Config={cfg['CONFIG']}"
                ]
                values = [[best_validation], [ep_test_auc], [ep]]
                append_or_create_excel(best_perf_filename, trial, identification, values)

                # Reset early stopping counter since validation improved
                with open(early_stopping_path, "w") as f:
                    f.write(str(cfg['PATIENCE']))

            # Early stopping condition: patience exhausted & minimum AUC threshold met
            with open(early_stopping_path, "r") as f:
                early_stopping_counter = int(f.read())

            if early_stopping_counter == -1 and ep_validation_auc >= 0.7 and ep_train_auc >= 0.7:
                break

        # Global early exit if training has already achieved nearly perfect AUC
        if ep_train_auc >= 0.99:
            break

    # Return tracked metric history after training completes
    return metrics


def read_yml(filepath: str) -> dict:
    """
    Load a YAML configuration file from disk and return its contents as a Python dictionary.

    Args:
        filepath (str): Path to the .yml file.

    Returns:
        dict: Parsed key-value pairs from the YAML file.
    """
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


def setup(rank: int, world_size: int) -> None:
    """
    Initialize the process group for Distributed Data Parallel (DDP) training.

    Args:
        rank (int): The index of the current process (corresponds to GPU id).
        world_size (int): Total number of processes/GPU devices involved in training.
    """

    # Set the master node's address (used for communication between processes)
    os.environ['MASTER_ADDR'] = 'localhost'

    # Assign a unique port for communication; handle SLURM-based port assignment if available
    try:
        # Use the last digits of the SLURM job ID to avoid port conflicts in multi-user environments
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]
        default_port = int(default_port) + 15000  # Offset to ensure it's above 10k
    except Exception:
        # Fallback to a static default port if SLURM_JOB_ID is not set
        default_port = 12910

    os.environ['MASTER_PORT'] = str(default_port)

    # Initialize the distributed training backend (NCCL is optimized for NVIDIA GPUs)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f'INITIALIZED RANK {rank}')


def cleanup():
    """
    Clean up the distributed process group after training ends.
    Ensures proper release of resources and communication channels.
    """
    dist.destroy_process_group()


def main() -> None:
    """
    Main entry point for launching the training script.
    Handles configuration loading, GPU setup, and spawning distributed processes.
    """

    # --------------------------------------------------------------------------
    # LOOP OVER SPLITS (can be extended to support multiple cross-validation runs)
    # --------------------------------------------------------------------------
    for SPLIT in [0]:
        # ----------------------------------------------------------------------
        # ARGUMENT PARSING
        # ----------------------------------------------------------------------
        parser = argparse.ArgumentParser()

        # Path to the configuration .yml file
        parser.add_argument(
            '-c', '--config',
            default='/home/hagen/Downloads/temp/config_2.yml',
            type=str,
            help="Path to the YAML config file"
        )

        # Optional: ID to identify which split/config is currently used
        parser.add_argument(
            "--config_id",
            type=int,
            required=False,
            help="Additional config ID (used for split assignment)"
        )

        args = parser.parse_args()
        print(args.config)  # Debug print for path verification

        # ----------------------------------------------------------------------
        # LOAD CONFIGURATION & SETUP
        # ----------------------------------------------------------------------
        config_id = args.config_id
        cfg = read_yml(args.config)  # Load configuration from YAML

        set_network_seed(cfg)  # Ensure deterministic training behavior

        # Assign current split to config for tracking & file naming
        cfg["SPLIT"] = config_id
        print(cfg)
        print("CONFIG ID:", config_id)

        # Change working directory to output path specified in config
        os.chdir(cfg["PATH_RESULT"])

        # Load additional model-specific settings (e.g., patch sizes)
        conf = models_genesis_config()

        # ----------------------------------------------------------------------
        # MULTI-GPU SETUP VIA DDP
        # ----------------------------------------------------------------------

        # Detect number of GPUs available for distributed training
        world_size = torch.cuda.device_count()

        # Spawn one process per GPU to run the `run()` function in parallel
        # Each process will independently handle training on a single GPU
        mp.spawn(
            run,
            args=(world_size, cfg, conf),
            nprocs=world_size
        )



if __name__ == '__main__':
    main()


