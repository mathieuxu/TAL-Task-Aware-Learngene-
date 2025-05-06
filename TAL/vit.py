# This file is used for TAL training, aligning TAL-generated ViT and teacher models.

from typing import Any, Optional
import torch
import torch.nn as nn
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models._api import WeightsEnum
from torchvision.models.vision_transformer import VisionTransformer as BaseVisionTransformer

class CustomVisionTransformer(BaseVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trans_in_dim = self.hidden_dim
        trans_out_dim = 768  # Hyperparameter for output dimension
        # Initialize additional member variables or components if needed
        self.encoder.trans = nn.Linear(trans_in_dim, trans_out_dim)

    def trans_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.encoder.trans(x)
        return x
    
def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    # representation_size: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> CustomVisionTransformer:
    """
    Build a CustomVisionTransformer for TAL training.
    """
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)
    model = CustomVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        # representation_size=representation_size,
        **kwargs,
    )
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model
