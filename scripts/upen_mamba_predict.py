#!/usr/bin/env python

import os
import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label

# Import model from the provided script
from utils_.UPEN_net_model import UPEN_mamba as Provided_UPEN_mamba  # Assuming UPEN_net_model.py is updated with your script

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

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions and visualizations using a trained UPEN_mamba model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--dataset', type=str, default="CHASEDB1", choices=["CHASEDB1", "DRIVE_DB"], help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader num_workers')
    parser.add_argument('--output-dir', type=str, default="outputs", help='Directory to save predictions and visualizations')
    parser.add_argument('--main-path', type=str, default="../datasets", help='Base path to datasets')
    args = parser.parse_args()
    return args

# Visualization functions
def compute_gradcam(model, image, layer, patch_size=96):
    model.eval()
    B, C, H, W = image.shape
    assert H >= patch_size and W >= patch_size, "Patch size must be smaller than image dimensions"

    h_start = (H - patch_size) // 2
    w_start = (W - patch_size) // 2
    patch = image[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size].clone().detach().to(torch.float16)

    features = None
    grads = None
    
    def feat_hook(module, input, output):
        nonlocal features
        features = output
    
    def grad_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]
    
    feat_handle = layer.register_forward_hook(feat_hook)
    grad_handle = layer.register_full_backward_hook(grad_hook)
    
    with torch.enable_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        out3, _, _ = model(patch)
        out3.mean().backward()
    
    weights = grads.mean(dim=(2, 3), keepdim=True).to(torch.float32)
    cam = F.relu((weights * features.to(torch.float32)).sum(dim=(0, 1))).detach().cpu().numpy()
    cam = cv2.resize(cam, (192, 192))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    feat_handle.remove()
    grad_handle.remove()
    return cam

def get_attention_map(model, image, layer):
    model.eval()
    attn_weights = None
    def hook(module, input, output):
        nonlocal attn_weights
        attn_weights = output  # Adjusted for 1x1 conv output
    handle = layer.attention.register_forward_hook(hook)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        model(image)
    handle.remove()
    # Since attention is now a 1x1 conv, use the output directly
    attn = attn_weights.mean(dim=1).cpu().numpy()  # [B, H, W]
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    return attn[0]  # Assuming batch size 1

def get_gradient_map(model, image):
    model.eval()
    image = image.clone().detach().to(torch.float16).requires_grad_(True)
    with torch.enable_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        out3, _, _ = model(image)
        out3.mean().backward()
    grad = image.grad.abs().sum(dim=(0, 1)).to(torch.float32).cpu().numpy()  # [H, W], e.g., [192, 192]
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return grad

def get_accurate_regions(pred_mask, gt_mask, iou_threshold=0.5):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    labeled_pred, num_features = label(pred_mask)
    print(f"Sample: Number of predicted regions = {num_features}")
    
    bboxes = []
    for i in range(1, num_features + 1):
        region_pred = (labeled_pred == i).astype(np.uint8)
        region_gt = gt_mask * region_pred
        intersection = np.sum(region_pred * region_gt)
        union = np.sum(region_pred) + np.sum(region_gt) - intersection
        iou = intersection / union if union > 0 else 0
        print(f"Region {i}: IoU = {iou:.4f}, Intersection = {intersection}, Union = {union}")
        if iou > iou_threshold:
            y, x = np.where(region_pred)
            if len(x) > 0 and len(y) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                bboxes.append((x_min, y_min, x_max, y_max))
                print(f"  Accurate region found: Bounding box = ({x_min}, {y_min}, {x_max}, {y_max})")
    print(f"Sample: Number of accurate regions = {len(bboxes)}")
    return bboxes

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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model parameters matching your trained UPEN_mamba
    in_channels = X_test.shape[-1]  # Assuming [N, H, W, C]
    features = [64, 128, 256, 512]  # From your model
    heads = 2
    num_dim = 64
    dropout_p = 0.2
    dilation = 2
    se_reduction = 16
    expand_type = 1
    function = 'arctan'

    # Load model
    model = Provided_UPEN_mamba(
        in_channels=in_channels,
        features=features,
        heads=heads,
        num_dim=num_dim,
        dropout_p=dropout_p,
        dilation=dilation,
        se_reduction=se_reduction,
        expand_type=expand_type,
        function=function
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    clean_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        clean_state_dict[new_key] = value
    model.load_state_dict(clean_state_dict)
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Inference and visualization
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Generating Predictions and Visualizations")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Get predictions (out3 is full resolution)
            out3, out2, out1 = model(images)
            probs = torch.sigmoid(out3).cpu().numpy()  # [B, 1, H, W]
            preds = (probs > 0.5).astype(np.uint8)     # [B, 1, H, W]

            # Process each sample in the batch
            for b in range(images.size(0)):
                sample_idx = idx * args.batch_size + b
                print(f"\nProcessing sample {sample_idx}")

                # Convert tensors to numpy arrays
                input_img = images[b].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                gt_mask = masks[b, 0].cpu().numpy()                     # [H, W]
                pred_mask = preds[b, 0]                                 # [H, W]
                prob_map = probs[b, 0]                                  # [H, W] for heatmap

                # Normalize input image for saving (0-255)
                input_img_uint8 = (input_img * 255).astype(np.uint8)

                # Save input image
                cv2.imwrite(os.path.join(args.output_dir, f'sample_{sample_idx}_input.png'), cv2.cvtColor(input_img_uint8, cv2.COLOR_RGB2BGR))

                # Save ground truth mask
                cv2.imwrite(os.path.join(args.output_dir, f'sample_{sample_idx}_gt.png'), (gt_mask * 255).astype(np.uint8))

                # Save predicted mask
                cv2.imwrite(os.path.join(args.output_dir, f'sample_{sample_idx}_pred.png'), (pred_mask * 255).astype(np.uint8))

                # Contour overlay visualization
                overlay = input_img_uint8.copy()
                gt_contours, _ = cv2.findContours((gt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pred_contours, _ = cv2.findContours((pred_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, gt_contours, -1, (0, 0, 255), 2)    # Blue for ground truth
                cv2.drawContours(overlay, pred_contours, -1, (0, 255, 0), 2)  # Green for prediction
                cv2.imwrite(os.path.join(args.output_dir, f'sample_{sample_idx}_contours.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                # Additional visualizations
                # 1. Grad-CAM (patch-based)
                torch.cuda.empty_cache()
                cam = compute_gradcam(model, images[b:b+1], model.decoder3.conv_block1, patch_size=96)
                plt.imshow(input_img, alpha=0.5)
                plt.imshow(cam, cmap='jet', alpha=0.5)
                plt.axis('off')
                plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx}_gradcam.png'), bbox_inches='tight')
                plt.close()

                # 2. Attention Map (from decoder3's attention layer)
                attn_map = get_attention_map(model, images[b:b+1], model.decoder3.conv_block1)
                plt.imshow(input_img, alpha=0.5)
                plt.imshow(attn_map, cmap='viridis', alpha=0.5)
                plt.axis('off')
                plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx}_attention.png'), bbox_inches='tight')
                plt.close()

                # 3. Gradient Activation Map
                grad_map = get_gradient_map(model, images[b:b+1])
                plt.imshow(input_img, alpha=0.5)
                plt.imshow(grad_map, cmap='hot', alpha=0.5)
                plt.axis('off')
                plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx}_gradmap.png'), bbox_inches='tight')
                plt.close()

                # 4. Probability Heatmap
                plt.imshow(input_img, alpha=0.5)
                plt.imshow(prob_map, cmap='jet', alpha=0.5)
                plt.axis('off')
                plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx}_probmap.png'), bbox_inches='tight')
                plt.close()

                # 5. Accurate Regions with Bounding Boxes
                pred_with_bboxes = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                bboxes = get_accurate_regions(pred_mask, gt_mask, iou_threshold=0.5)
                for (x_min, y_min, x_max, y_max) in bboxes:
                    cv2.rectangle(pred_with_bboxes, (x_min, y_min), (x_max, y_max), (144, 238, 144), 2)  # Light green
                cv2.imwrite(os.path.join(args.output_dir, f'sample_{sample_idx}_accurate_regions.png'), pred_with_bboxes)

    print(f"Predictions and visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()
