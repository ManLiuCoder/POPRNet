from .POPRNet import build_POPRNet

_ZSL_META_ARCHITECTURES = {
    "Model": build_POPRNet,
}

def build_zsl_pipeline(cfg):
    meta_arch = _ZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)