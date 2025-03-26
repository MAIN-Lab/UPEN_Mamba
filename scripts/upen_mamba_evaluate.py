#!/usr/bin/env python

import os
import numpy as np
import argparse
from datetime import datetime
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ptflops import get_model_complexity_info  # For FLOPs and parameter count

# Import model from the utils_ folder
from utils_.UPEN_net_model import UPEN_mamba

# Dataset class (same as training script, without augmentation)
class MedicalImageDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images  # shape [N,H,W,C]
        self.masks = masks    # shape [N,H,W] or [N,H,W,1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, C)
        mask = self.masks[idx]    # (H, W) or (H,W,1)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # => (C,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()       # => (1,H,W)
        return image, mask

# Load dataset (same as training script)
def load_dataset(dataset_name, dim, main_path):
    testing_images = f"{main_path}/{dataset_name}/patches_{dim}/images/test/"
    testing_labels = f"{main_path}/{dataset_name}/patches_{dim}/labels/test/"

    test_img_files = sorted(os.listdir(testing_images))
    test_lbl_files = sorted(os.listdir(testing_labels))
    X_test = np.concatenate([np.load(os.path.join(testing_images, file))['arr_0'] for file in test_img_files], axis=0)
    y_test = np.concatenate([np.load(os.path.join(testing_labels, file))['arr_0'] for file in test_lbl_files], axis=0)

    X_test = X_test / 255.0
    if dataset_name == "CHASEDB1":
        y_test = y_test.astype('float32')
    else:
        y_test = y_test / 255.0

    return X_test, y_test

# Evaluation metrics per sample (only Dice)
def compute_sample_metrics(preds, targets, threshold=0.5, eps=1e-8):
    preds = (preds > threshold).astype(np.uint8)  # Binary predictions
    targets = (targets > 0).astype(np.uint8)      # Binary ground truth

    intersection = np.sum(preds * targets)
    dice = (2. * intersection + eps) / (np.sum(preds) + np.sum(targets) + eps)

    return dice

# Compute total metrics over all samples (Dice, Recall, Precision, Accuracy)
def compute_total_metrics(all_preds, all_targets, threshold=0.5, eps=1e-8):
    all_preds = (all_preds > threshold).astype(np.uint8)
    all_targets = (all_targets > 0).astype(np.uint8)

    # True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = np.sum(all_preds * all_targets)
    fp = np.sum(all_preds * (1 - all_targets))
    fn = np.sum((1 - all_preds) * all_targets)
    tn = np.sum((1 - all_preds) * (1 - all_targets))

    # Dice Score
    dice = (2. * tp + eps) / (np.sum(all_preds) + np.sum(all_targets) + eps)

    # Recall (Sensitivity)
    recall = (tp + eps) / (tp + fn + eps)

    # Precision
    precision = (tp + eps) / (tp + fp + eps)

    # Accuracy
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return dice, recall, precision, accuracy

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained UPEN_mamba model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model .pth file in models/ folder')
    parser.add_argument('--main-path', type=str, default="../datasets", help='Base path to datasets')
    parser.add_argument('--dataset', type=str, default="CHASEDB1", choices=["CHASEDB1", "DRIVE_DB"], help='Dataset name')
    parser.add_argument('--num-dim', type=int, default=64, help='Number of dimensions in attention block')
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128, 256, 512], help='List of feature channels (e.g., --features 64 128 256 512)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--dilation', type=int, default=2, help='Dilation factor for convolutions')
    parser.add_argument('--se-reduction', type=int, default=16, help='Reduction ratio for SE')
    parser.add_argument('--expand-type', type=int, default=1, help='Expansion type for ProgressiveExpansionQKV')
    parser.add_argument('--function', type=str, default='arctan', choices=['arctan', 'other_function'], help='Function for ProgressiveExpansionQKV')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device setup (use GPU 0 explicitly)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load test data
    X_test, y_test = load_dataset(args.dataset, dim=192, main_path=args.main_path)
    test_dataset = MedicalImageDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Model parameters
    in_channels = X_test.shape[-1]  # Assuming [N, H, W, C]
    features = args.features
    num_dim = args.num_dim
    dropout_p = args.dropout
    dilation = args.dilation
    se_reduction = args.se_reduction
    expand_type = args.expand_type
    function = args.function

    # Load model
    model = UPEN_mamba(
        in_channels=in_channels,
        features=features,
        num_dim=num_dim,
        dropout_p=dropout_p,
        dilation=dilation,
        se_reduction=se_reduction,
        expand_type=expand_type,
        function=function
    ).to(device)

    # Ensure model path is correctly formatted
    model_path = os.path.join("models", args.model_path) if not args.model_path.startswith("models/") else args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    clean_state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in checkpoint.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # Compute FLOPs and parameters
    input_shape = (in_channels, 192, 192)  # Assuming 192x192 patches
    flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False)
    print(f"Computational Complexity: {flops}")
    print(f"Number of Parameters: {params}")

    # Initialize accumulators
    num_samples = 0
    all_preds = []
    all_targets = []
    inference_times = []

    # Memory tracking setup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB

    # Inference and evaluation
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating Model")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Measure inference time
            start_time = time.time()
            out3, _, _ = model(images)
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            probs = torch.sigmoid(out3).cpu().numpy()  # [B, 1, H, W]
            gt_masks = masks.cpu().numpy()             # [B, 1, H, W]

            # Current sample (batch size is 1)
            pred = probs[0, 0]       # [H, W]
            target = gt_masks[0, 0]  # [H, W]
            dice = compute_sample_metrics(pred, target)

            num_samples += 1

            # Accumulate predictions and targets for total metrics
            all_preds.append(pred)
            all_targets.append(target)

            print(f"Sample {idx}: Dice Score = {dice:.4f}, Inference Time = {inference_time:.4f} sec")

    # Compute average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)

    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # Convert to MB
        memory_used = peak_memory - initial_memory
    else:
        peak_memory = "N/A (CPU)"
        memory_used = "N/A (CPU)"

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)    # Shape: [N*H*W]
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: [N*H*W]

    # Compute total metrics
    dice, recall, precision, accuracy = compute_total_metrics(all_preds, all_targets)

    # Print final evaluation results
    print("\nEvaluation Results:")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Features: {features}")
    print(f"Num_dim: {num_dim}")
    print(f"Dice Score: {dice:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nComputational Efficiency Metrics:")
    print(f"Average Inference Time per Sample: {avg_inference_time:.4f} sec")
    print(f"Peak GPU Memory Usage: {peak_memory} MB")
    print(f"Memory Consumption (Model + Data): {memory_used} MB")
    print(f"Computational Complexity: {flops}")
    print(f"Number of Parameters: {params}")

if __name__ == '__main__':
    main()
