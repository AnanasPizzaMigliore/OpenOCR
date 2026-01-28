import torch
import torch.nn as nn
import timm
import logging
import numpy as np
from openrec.modeling.common import Block  # uses your framework's transformer block

logger = logging.getLogger("openrec")

class TIMMTrans(nn.Module):
    def __init__(
        self,
        model_name='resnet50',
        pretrained=True,
        in_channels=3,
        out_indices=(1, 2, 3, 4),
        out_channels=256,
        trans_layer=0,
        out_dim=384,
        drop_path_rate=0.1,
        last_stage=True,
    ):
        super().__init__()
        logger.info(f"[TIMMWrapper] Creating model='{model_name}', pretrained={pretrained}, out_indices={out_indices}")

        # 1. Build timm CNN backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=out_indices
        )

        # 2. Output info
        self.out_channels_list = self.backbone.feature_info.channels()
        self.out_channels = self.out_channels_list[-1]

        # 3. Projection layer to align channel dims
        if out_channels and out_channels != self.out_channels:
            self.proj = nn.Conv2d(self.out_channels, out_channels, 1)
            self.out_channels = out_channels
        else:
            self.proj = nn.Identity()

        # 4. Optional transformer refinement
        self.trans_layer = trans_layer
        if trans_layer > 0:
            logger.info(f"[TIMMWrapper] Adding {trans_layer} Transformer layers (dim={out_dim})")
            dpr = np.linspace(0, drop_path_rate, trans_layer)
            blocks = [
                nn.Linear(out_channels, out_dim)
            ] + [
                Block(
                    dim=out_dim,
                    num_heads=max(1, out_dim // 32),
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop_path=dpr[i]
                )
                for i in range(trans_layer)
            ]
            self.trans_blocks = nn.Sequential(*blocks)
            self.out_channels = out_dim
        else:
            self.trans_blocks = None

        # 5. Optional final sequence projection
        self.last_stage = last_stage
        if last_stage:
            self.last_conv = nn.Linear(self.out_channels, out_channels)
            self.dropout = nn.Dropout(0.1)
            self.hardswish = nn.Hardswish()
            self.out_channels = out_channels

        logger.info(f"[TIMMWrapper] Final out_channels = {self.out_channels}")

    def forward(self, x):
        feats = self.backbone(x)
        x = feats[-1]  # last feature map
        x = self.proj(x)

        if self.trans_blocks is not None:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            x = self.trans_blocks(x)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        if self.last_stage:
            x = x.mean(2).transpose(1, 2)  # (B, W, C)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)

        return x
