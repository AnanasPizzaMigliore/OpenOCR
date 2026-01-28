# timm_wrapper.py
import torch
import torch.nn as nn
from timm import create_model
from tools.utils.logging import get_logger

logger = get_logger()


class TIMMWrapper(nn.Module):
    """
    TIMM backbone wrapper that matches RepSVTR_det I/O:
      - Input:  x -> [B, 3, H, W]
      - Output: list of feature maps [f1, f2, f3, f4] (for FPN)
    Exposes:
      - self.out_channels: [C1, C2, C3, C4]
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        out_indices=(1, 2, 3, 4),
        freeze_stages: int = -1,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_indices = tuple(out_indices)

        logger.info(
            f"[TIMMWrapper] Creating model='{model_name}', "
            f"pretrained={pretrained}, in_chans={in_channels}, out_indices={self.out_indices}"
        )

        # Create a features-only TIMM model that returns a list of feature maps
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,          # <-- important
            out_indices=self.out_indices,
            in_chans=in_channels,
        )

        # Optionally freeze early stages
        if freeze_stages >= 0 and hasattr(self.backbone, "stages"):
            for i, stage in enumerate(self.backbone.stages):
                if i <= freeze_stages:
                    for p in stage.parameters():
                        p.requires_grad = False
                    logger.info(f"[TIMMWrapper] Froze stage {i}")
        elif freeze_stages >= 0:
            logger.info("[TIMMWrapper] Model has no 'stages' attr; skipping freezing.")

        # --- Determine out_channels for FPN/neck consumers ---
        self.out_channels = self._infer_out_channels()
        logger.info(f"[TIMMWrapper] out_channels = {self.out_channels}")

    @torch.no_grad()
    def _infer_out_channels(self):
        # Prefer TIMM's feature_info if available
        if hasattr(self.backbone, "feature_info") and hasattr(self.backbone.feature_info, "channels"):
            chans = list(self.backbone.feature_info.channels())  # already aligned with out_indices
            if isinstance(chans[0], (list, tuple)):
                chans = [int(c[0]) for c in chans]  # just in case some models wrap it
            return [int(c) for c in chans]

        # Fallback: do a tiny dummy forward on CPU to get shapes
        logger.warning("[TIMMWrapper] feature_info.channels() not available; using dummy forward to infer channels.")
        dev = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, 3, 320, 320, device=dev)  # smaller to be safe
        feats = self.backbone(dummy)
        return [int(f.shape[1]) for f in feats]

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)  # list[Tensor]
        return feats
