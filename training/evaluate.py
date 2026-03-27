"""
Evaluation utilities for computing classification metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run inference over a DataLoader and return loss + classification metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(y)
            all_preds.extend((pred.sigmoid() >= 0.5).cpu().long().tolist())
            all_labels.extend(y.cpu().long().tolist())

    n = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": total_loss / n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
