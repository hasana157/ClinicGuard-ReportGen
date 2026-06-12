"""
Main training script for the pathology classification model.

Performs deterministic initialization, loads dataset splits, initializes the
GroundedVisionEncoder, sets up BCE loss and AdamW, runs the training/validation epoch loops,
applies early stopping, and saves the best model checkpoint.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config, ProjectConfig, PATHOLOGY_LABELS
from src.data_loader import get_dataloaders
from src.preprocessing import get_train_transforms, get_eval_transforms, binarize_labels
from src.vision_encoder import GroundedVisionEncoder


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        # Binarize label targets (-1.0 uncertainty converted to 0.0 for BCE)
        labels = binarize_labels(batch["labels"], uncertain_policy="zero").to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output["logits"], labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = criterion(output["logits"], labels)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            labels = binarize_labels(batch["labels"], uncertain_policy="zero").to(device)
            
            output = model(images)
            loss = criterion(output["logits"], labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Save predictions and targets
            probs = output["probabilities"].cpu().numpy()
            all_preds.append(probs)
            all_trues.append(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    
    # Calculate simple validation accuracy metric
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    preds_binary = (all_preds >= 0.5).astype(float)
    acc = np.mean(preds_binary == all_trues)
    
    return val_loss, {"accuracy": acc}


def plot_curves(train_losses: List[float], val_losses: List[float], save_path: str):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue", lw=2)
    plt.plot(val_losses, label="Val Loss", color="orange", lw=2)
    plt.title("Pathology Classification Model Loss Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curves plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pathology Classification training script")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone weights")
    
    args = parser.parse_args()
    
    # 1. Setup config and directories
    config = get_config()
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.batch_size = args.batch_size
    config.training.seed = args.seed
    config.vision.freeze_backbone = args.freeze
    
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.evaluation.checkpoint_dir if hasattr(config.evaluation, 'checkpoint_dir') else config.training.checkpoint_dir, exist_ok=True)
    
    set_seed(config.training.seed)
    
    device = torch.device(args.device)
    print(f"Training on device: {device}")
    
    # 2. Get Transforms and Dataloaders
    train_transform = get_train_transforms(config.vision.input_size)
    eval_transform = get_eval_transforms(config.vision.input_size)
    
    train_loader, val_loader, _ = get_dataloaders(
        config=config,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    
    print(f"Dataset Loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 3. Create Model
    model = GroundedVisionEncoder(config=config.vision)
    model.to(device)
    
    # 4. Define Loss, Optimizer and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # 5. Training Loop
    train_losses = []
    val_losses = []
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    print("Starting Training Loop...")
    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch Summary: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Accuracy = {metrics['accuracy']:.1%}")
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"🔥 Best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}.")
                break
                
    # Plot losses
    curves_path = os.path.join(config.training.checkpoint_dir, "loss_curves.png")
    plot_curves(train_losses, val_losses, curves_path)
    
    print("Training finished! Best validation loss achieved:", best_val_loss)


if __name__ == "__main__":
    main()
