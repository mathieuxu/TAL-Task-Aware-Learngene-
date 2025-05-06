import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer


def load_teachermodel(model_path, num_classes=100):    
    # Load ViT-base model (not pretrained by default)
    teachermodel = timm.create_model("vit_base_patch16_224", pretrained=False)

    # Replace the classification head for the desired number of classes
    teachermodel.head = nn.Linear(teachermodel.head.in_features, num_classes)

    # Load fine-tuned weights
    teachermodel.load_state_dict(torch.load(model_path))
    print('Teachermodel successfully loaded...')
 
    return teachermodel


def modify_teacher_head(teachermodel, teacher_dim=192, num_classes=100):
    """
    Add an MLP layer to the teacher model's head for producing soft labels for distillation.

    Args:
        teachermodel (nn.Module): The teacher model to modify.
        teacher_dim (int): The intermediate dimension for the MLP.
        num_classes (int): Number of output classes.
    """
    # Build a new classification head
    new_head = nn.Sequential(
        nn.Linear(768, teacher_dim),
        nn.Linear(teacher_dim, num_classes)
    )
    # Replace the model's head
    teachermodel.head = new_head


def get_teacher_label(teachermodel, inputs, is_imagenet=True):
    """
    Obtain soft labels of dimension teacher_dim from the teacher model for loss computation.

    Args:
        teachermodel (nn.Module): The teacher model.
        inputs (Tensor): Input images.
        is_imagenet (bool): Whether the dataset is ImageNet (default: True).

    Returns:
        Tensor: Soft label features of shape (batch_size, teacher_dim).
    """
    # Forward pass. The process may differ depending on the teacher model.
    # For ViT-base (timm.create_model("vit_base_patch16_224", pretrained=False)), see:
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L719
    x = teachermodel.forward_features(inputs)
    x = teachermodel.fc_norm(x)
    x = teachermodel.head_drop(x)
    # If not ImageNet, apply the first layer of the head
    if not is_imagenet:
        x = teachermodel.head[0](x)
    x = x[:, 0, :]  # Take the first token's features
    return x