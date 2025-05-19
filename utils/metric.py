import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_accuracy(predicted, target):
    _, predicted_labels = torch.max(predicted, dim=1)
    _, target_labels = torch.max(target, dim=1)
    correct_pixels = torch.sum(predicted_labels == target_labels).item()
    total_pixels = target_labels.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy


def intersection_over_union(predicted, target, num_classes=2):
    _, predicted_labels = torch.max(predicted, dim=1)
    _, target_labels = torch.max(target, dim=1)
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for cls in range(1, num_classes):
        pred_cls = (predicted_labels == cls)
        target_cls = (target_labels == cls)
        intersection[cls] = torch.sum(pred_cls & target_cls)
        union[cls] = torch.sum(pred_cls | target_cls)
    iou_per_class = intersection[1:] / (union[1:] + 1e-8)
    iou = torch.mean(iou_per_class)
    return iou.item()

