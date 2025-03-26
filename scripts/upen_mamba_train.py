#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from tqdm import tqdm

# Import UPEN_mamba
from utils_.UPEN_net_model import UPEN_mamba

# Dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images = images  # shape [N,H,W,C]
        self.masks = masks    # shape [N,H,W] or [N,H,W,1]
        self.augment = augment

        self.transform_color = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2)])
        self.transform_geo = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, C)
        mask  = self.masks[idx]   # (H, W) or (H,W,1)
        image = torch.from_numpy(image).permute(2,0,1).float()  # => (C,H,W)
        mask  = torch.from_numpy(mask).unsqueeze(0).float()     # => (1,H,W)

        if self.augment:
            image = self.transform_color(image)
            stacked = torch.cat([image, mask], dim=0)  # (C+1, H, W)
            stacked = self.transform_geo(stacked)
            image = stacked[:-1]
            mask  = stacked[-1:].clamp(0, 1)
        return image, mask

# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.eps) / (inputs.sum() + targets.sum() + self.eps)
        return 1 - dice

def dice_bce_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = DiceLoss()(pred, target)
    return bce + dice

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-7):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky_index = (TP + self.eps) / (TP + self.alpha*FN + self.beta*FP + self.eps)
        focal_tversky = (1 - tversky_index) ** self.gamma
        return focal_tversky

def lovasz_hinge(logits, labels):
    logits, labels = logits.view(-1), labels.view(-1)
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(dim=0)
    union = gts + (1 - gt_sorted).cumsum(dim=0)
    jaccard = 1. - intersection / union
    grad = jaccard.clone()
    grad[1:] = grad[1:] - grad[:-1]
    return grad

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        return lovasz_hinge(logits, targets)

class ContourAwareLoss(nn.Module):
    def __init__(self, alpha=0.1, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss(eps=self.eps)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice_fn(logits, targets)
        preds = torch.sigmoid(logits)
        pred_edges = edge_map(preds)
        true_edges = edge_map(targets)
        contour_loss = F.l1_loss(pred_edges, true_edges)
        return bce_loss + dice_loss + self.alpha * contour_loss

def edge_map(tensor):
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=tensor.device).float()
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=tensor.device).float()
    gx = F.conv2d(tensor, sobel_x, padding=1)
    gy = F.conv2d(tensor, sobel_y, padding=1)
    grad = torch.sqrt(gx**2 + gy**2 + 1e-6)
    return grad

def get_loss_function(loss_name):
    if loss_name == 'dice_bce':
        return dice_bce_loss
    elif loss_name == 'focal_tversky':
        return FocalTverskyLoss()
    elif loss_name == 'lovasz':
        return LovaszHingeLoss()
    elif loss_name == 'contour_aware':
        return ContourAwareLoss(alpha=0.1)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

# Load Data
def load_dataset(dataset_name, dim, main_path):
    training_images = f"{main_path}/{dataset_name}/patches_{dim}/images/train/"
    training_labels = f"{main_path}/{dataset_name}/patches_{dim}/labels/train/"
    testing_images  = f"{main_path}/{dataset_name}/patches_{dim}/images/test/"
    testing_labels  = f"{main_path}/{dataset_name}/patches_{dim}/labels/test/"

    train_img_files = sorted(os.listdir(training_images))
    train_lbl_files = sorted(os.listdir(training_labels))
    X_train = np.concatenate([np.load(os.path.join(training_images, file))['arr_0'] for file in train_img_files], axis=0)
    y_train = np.concatenate([np.load(os.path.join(training_labels, file))['arr_0'] for file in train_lbl_files], axis=0)

    test_img_files = sorted(os.listdir(testing_images))
    test_lbl_files = sorted(os.listdir(testing_labels))
    X_test = np.concatenate([np.load(os.path.join(testing_images, file))['arr_0'] for file in test_img_files], axis=0)
    y_test = np.concatenate([np.load(os.path.join(testing_labels, file))['arr_0'] for file in test_lbl_files], axis=0)

    X_train = X_train / 255.0
    X_test  = X_test  / 255.0
    if dataset_name == "CHASEDB1":
        y_train = y_train.astype('float32')
        y_test  = y_test.astype('float32')
    else:
        y_train = y_train / 255.0
        y_test  = y_test  / 255.0

    return X_train, y_train, X_test, y_test

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train UPEN_mamba model.")
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader num_workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--loss', type=str, default='dice_bce',
                        choices=['dice_bce', 'focal_tversky', 'lovasz', 'contour_aware'],
                        help='Loss function to use')
    parser.add_argument('--dataset', type=str, default="DRIVE_DB",
                        choices=["CHASEDB1", "DRIVE_DB", "Full"],
                        help='Dataset name')
    parser.add_argument('--num-dim', type=int, default=64, help='Projection dimension for attention')
    parser.add_argument('--features', type=str, default="[64, 128, 256, 512]",
                        help='Feature map sizes, e.g. "[64, 128, 256, 512]"')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--dilation', type=int, default=2, help='Dilation factor for convolutions')
    parser.add_argument('--se-reduction', type=int, default=16, help='Reduction ratio for SE')
    
    args = parser.parse_args()
    try:
        args.features = eval(args.features)
        if not isinstance(args.features, list) or not all(isinstance(x, int) for x in args.features):
            raise ValueError
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("--features must be a list of ints, e.g. '[64, 128, 256, 512]'")
    return args

# Main Training Function
def main():
    args = parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = args.dataset
    dim = 192
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_workers = args.num_workers
    lr = args.lr
    weight_decay = args.weight_decay
    loss_name = args.loss
    num_dim = args.num_dim
    features = args.features
    dropout_p = args.dropout
    dilation = args.dilation
    se_reduction = args.se_reduction
    main_path = "../datasets"

    if dataset == "Full":
        X_train_chase, y_train_chase, X_test_chase, y_test_chase = load_dataset("CHASEDB1", dim, main_path)
        X_train_drive, y_train_drive, X_test_drive, y_test_drive = load_dataset("DRIVE_DB", dim, main_path)
        X_train = np.concatenate((X_train_chase, X_train_drive), axis=0)
        y_train = np.concatenate((y_train_chase, y_train_drive), axis=0)
        test_sets = {"CHASEDB1": (X_test_chase, y_test_chase), "DRIVE_DB": (X_test_drive, y_test_drive)}
        X_val, y_val = X_test_chase, y_test_chase
    else:
        X_train, y_train, X_test, y_test = load_dataset(dataset, dim, main_path)
        test_sets = {dataset: (X_test, y_test)}
        X_val, y_val = X_test, y_test

    train_dataset = MedicalImageDataset(X_train, y_train, augment=True)
    val_dataset   = MedicalImageDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    _, _, _, c = X_train.shape

    # Instantiate UPEN_mamba model
    model = UPEN_mamba(
        in_channels=c,
        features=features,
        num_dim=num_dim,
        dropout_p=dropout_p,
        dilation=dilation,
        se_reduction=se_reduction
    )
    model_name = f"UPEN_mamba_{dataset}_best_model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")

    # Loss, optimizer, scheduler
    loss_fn = get_loss_function(loss_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_loss = float('inf')
    history = {'loss': [], 'val_loss': []}

    # Training
    start_time = datetime.now()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", unit="batch")
        for images, masks in train_pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                out3, out2, out1 = model(images)
                mask_full = masks
                mask_half = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=True)
                mask_quarter = F.interpolate(masks, scale_factor=0.25, mode='bilinear', align_corners=True)
                loss = (loss_fn(out3, mask_full) + 
                        0.5 * loss_fn(out2, mask_half) + 
                        0.25 * loss_fn(out1, mask_quarter))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", unit="batch")
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)
                with autocast():
                    out3, out2, out1 = model(images)
                    mask_full = masks
                    mask_half = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=True)
                    mask_quarter = F.interpolate(masks, scale_factor=0.25, mode='bilinear', align_corners=True)
                    v_loss = (loss_fn(out3, mask_full) + 
                              0.5 * loss_fn(out2, mask_half) + 
                              0.25 * loss_fn(out1, mask_quarter))
                val_loss += v_loss.item()
                val_pbar.set_postfix({"val_loss": f"{v_loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/{model_name}")
            print(f"Model saved with val loss {best_loss:.4f}")

    execution_time = datetime.now() - start_time
    print(f"Training completed in: {execution_time}")

    # Save history
    os.makedirs("results", exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'results/{model_name}_history.csv', index=False)

    # Plot learning curves
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/learning_curve_UPEN_mamba_dataset_{args.dataset}_loss_{args.loss}.png"
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve (UPEN_mamba, {args.dataset})')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Learning curve saved to {plot_filename}")

    # Evaluation
    model_eval = UPEN_mamba(
        in_channels=c,
        features=features,
        num_dim=num_dim,
        dropout_p=dropout_p,
        dilation=dilation,
        se_reduction=se_reduction
    ).to(device)

    checkpoint = torch.load(f"models/{model_name}", map_location=device, weights_only=True)
    clean_state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in checkpoint.items()}
    model_eval.load_state_dict(clean_state_dict)
    model_eval.eval()

    results_text = [f"Model execution time: {execution_time}\n"]

    for ds_name, (X_test_ds, y_test_ds) in test_sets.items():
        test_dataset = MedicalImageDataset(X_test_ds, y_test_ds, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        all_preds = []
        all_labels = []

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, masks in test_loader:
                images = images.to(device, non_blocking=True)
                logits, _, _ = model_eval(images)  # Use only out3
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = masks.cpu().numpy()

                preds = (probs > 0.5).astype(np.uint8)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = np.concatenate(all_preds, axis=0).reshape(-1)
        all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

        precision = precision_score(all_labels, all_preds)
        recall    = recall_score(all_labels, all_preds)
        accuracy  = accuracy_score(all_labels, all_preds)
        dice      = f1_score(all_labels, all_preds)

        ds_result = (
            f"Dataset: {ds_name}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Dice Score: {dice:.4f}\n"
        )
        print(ds_result)
        results_text.append(ds_result)

    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_metrics.txt", 'w') as f:
        f.write("\n".join(results_text))

if __name__ == '__main__':
    main()
