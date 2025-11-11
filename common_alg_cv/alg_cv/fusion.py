import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval


def _is_conv(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))

def _is_bn(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

def _can_fuse_pair(a: nn.Module, b: nn.Module) -> bool:
    # Both must be eval(); BN must track running stats and have them populated
    if a.training or b.training:
        return False
    if _is_conv(a) and _is_bn(b):
        bn_ok = getattr(b, "track_running_stats", True) and b.running_mean is not None and b.running_var is not None
        return bn_ok
    if isinstance(a, nn.Linear) and isinstance(b, nn.BatchNorm1d):
        bn_ok = getattr(b, "track_running_stats", True) and b.running_mean is not None and b.running_var is not None
        return bn_ok
    return False

def _fuse_pair(a: nn.Module, b: nn.Module) -> nn.Module:
    if _is_conv(a) and _is_bn(b):
        return fuse_conv_bn_eval(a, b)
    if isinstance(a, nn.Linear) and isinstance(b, nn.BatchNorm1d):
        return fuse_linear_bn_eval(a, b)
    return a  # shouldn't happen if guarded by _can_fuse_pair

def fuse_model_inplace(model: nn.Module) -> nn.Module:
    """
    Recursively fuses Conv/Linear + BN pairs in-place.
    Replaces the BN with nn.Identity().
    Only eval-mode pairs are fused; others are left unchanged.
    """
    # Work bottom-up
    for name, child in list(model.named_children()):
        fuse_model_inplace(child)

    # Now try to fuse adjacent children in this module
    # We need ordered traversal/edit of _modules
    keys = list(model._modules.keys())
    i = 0
    while i < len(keys) - 1:
        k1, k2 = keys[i], keys[i+1]
        m1, m2 = model._modules[k1], model._modules[k2]

        if _can_fuse_pair(m1, m2):
            fused = _fuse_pair(m1, m2)
            model._modules[k1] = fused
            model._modules[k2] = nn.Identity()
            # advance by 2 to avoid re-checking the identity just inserted
            i += 2
        else:
            i += 1
    return model
