# Image loader for multitask

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from teachermodels import * 

# The following code does not remap the classification head of the image model

def create_task_dataset(dataset, class_indices):
    """
    Create a subset of the dataset for a task based on specified class indices.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        class_indices (list of int): Class indices for the task.

    Returns:
        torch.utils.data.Subset: Subset dataset containing only the specified classes.
    """
    targets = np.array(dataset.targets)
    task_indices = np.isin(targets, class_indices).nonzero()[0]
    return Subset(dataset, task_indices)


def get_task_embedding(model, dataloader, device, is_imagenet=False):
    """
    Compute the mean feature embedding for a task using the teacher model.

    Args:
        model (nn.Module): The teacher model.
        dataloader (DataLoader): DataLoader for the task.
        device (torch.device): Device to run the model on.
        is_imagenet (bool): Whether the dataset is ImageNet.

    Returns:
        torch.Tensor: The mean feature embedding for the task.
    """
    model.eval()
    model.to(device)
    running_mean = 0
    total_samples = 0
    with torch.no_grad():
        print(len(dataloader))
        for images, targets in dataloader:
            images = images.to(device)
            outputs = get_teacher_label(model, images, is_imagenet=is_imagenet)
            batch_mean = outputs.mean(dim=0)
            batch_size = outputs.size(0)
            running_mean += batch_mean * batch_size
            total_samples += batch_size
    task_embedding = running_mean / total_samples
    return task_embedding.cpu()


def map_labels(labels, label_map):
    """
    Map original labels to new labels using a label map.

    Args:
        labels (torch.Tensor): Original labels.
        label_map (dict): Mapping from original label to new label.

    Returns:
        torch.Tensor: Mapped labels.
    """
    mapped_labels = torch.zeros_like(labels)
    for i in range(labels.size(0)):
        label = labels[i].item()
        mapped_labels[i] = label_map[label]
    return mapped_labels