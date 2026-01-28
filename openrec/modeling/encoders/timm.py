import torch
import torch.nn as nn
from timm import create_model


class TIMMEncoder(nn.Module):
    """
    Unified timm encoder for OCR (CNN/ViT/hybrid).
    Compatible with OpenOCR config keys: in_channels, out_dim, out_indices.
    """

    def __init__(
        self,
        model_name="repvit_m1.dist_in1k",
        pretrained=True,
        in_channels=3,
        out_dim=256,
        out_channels=None,
        out_indices=(-1,),
        **kwargs
    ):
        super().__init__()

        # unify arg names
        out_channels = out_channels or out_dim or 256

        # create timm backbone
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,   # timm param name
        )

        in_chs = self.backbone.feature_info[out_indices[-1]]["num_chs"]
        self.proj = nn.Conv2d(in_chs, out_channels, 1)

        self.model_name = model_name
        self.out_channels = out_channels
        self.out_indices = out_indices

    def forward(self, x):
        feats = self.backbone(x)[-1]   # take the last feature map
        feats = self.proj(feats)
        return feats

    def __repr__(self):
        return f"TIMMEncoder(model={self.model_name}, out={self.out_channels}, indices={self.out_indices})"
