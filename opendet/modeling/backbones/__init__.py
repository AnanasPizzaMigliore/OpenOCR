__all__ = ['build_backbone']

def build_backbone(config):
    from .repvit import RepSVTR_det
    from .timm_wrapper import TIMMWrapper

    registry = {
        'RepSVTR_det': RepSVTR_det,
        'TIMMWrapper': TIMMWrapper,
    }

    name = config.pop('name')
    if name not in registry:
        raise ValueError(f"[Backbone] Unsupported backbone '{name}'. "
                         f"Available options: {list(registry.keys())}")

    cls = registry[name]
    backbone = cls(**config)
    return backbone
