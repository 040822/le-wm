"""Opt-in implementation variants for the Phase 6.3 experiments.

These adapters patch only the experiment's model instance and can be removed.
Defaults and archived Table 1 implementations are unchanged.
"""
from contextlib import contextmanager
from types import MethodType

import torch


class FixedProjectionCache:
    """Cache frozen projections of unchanged tensor views within one replan.

    Keys include storage address, offset, shape, strides, dtype/device and
    mutation version. Strong references prevent allocator address reuse. Views
    of different candidate/selected rows never alias merely by storage address.
    Inputs with gradients bypass the cache, preserving action derivatives.
    """

    def __init__(self):
        self.values = {}
        self.hits = 0
        self.misses = 0
        self.replans = 0

    def clear(self):
        self.values.clear()
        self.replans += 1

    def project(self, name, original, value):
        if value.requires_grad:
            return original(value)
        try:
            version = value._version
        except RuntimeError:
            # Inference-mode tensors have no mutation counter: do not cache.
            return original(value)
        key = (name, value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()), value.dtype, value.device,
               version, torch.is_autocast_enabled(value.device.type))
        entry = self.values.get(key)
        if entry is not None:
            self.hits += 1
            return entry[1]
        result = original(value)
        self.values[key] = (value, result)
        self.misses += 1
        return result


@contextmanager
def fixed_condition_cache(model):
    """Cache z-condition and anchor projections; clear at every image encoding.

    Every supported policy rebuilds current latents at each real replan. Clearing
    before encoding therefore includes invalidation cost in the full call and
    never carries fixed projections or goal latents across replans. A second
    encode in the same replan clears conservatively, rather than reusing a stale
    context. Frozen parameters are mandatory; dependent action projections and
    timestep/AdaLN outputs without an unambiguous reuse identity remain native.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("fixed condition cache requires frozen parameters")
    cache = FixedProjectionCache()
    restoration = []

    def replace(owner, name, replacement):
        had = name in owner.__dict__
        saved = owner.__dict__.get(name)
        restoration.append((owner, name, had, saved))
        setattr(owner, name, MethodType(replacement, owner))

    for name in ("z_condition", "latent_input"):
        module = getattr(model, name)
        original = module.forward

        def forward(_self, value, _name=name, _original=original):
            return cache.project(_name, _original, value)

        replace(module, "forward", forward)
    original_encode = model.encode_pixels

    def encode(_self, pixels):
        cache.clear()
        return original_encode(pixels)

    replace(model, "encode_pixels", encode)
    try:
        yield cache
    finally:
        cache.values.clear()
        for owner, name, had, saved in reversed(restoration):
            if had:
                setattr(owner, name, saved)
            else:
                delattr(owner, name)


@contextmanager
def preprocess_optimized(policy):
    """Apply the native image pipeline to its existing image batch in one call.

    The image conversion, normalization and resize order/parameters stay native.
    All non-image processing retains BasePolicy's behavior. This eliminates
    per-frame Python dispatch and the final stack copy, without moving any work
    out of the policy call, changing scoring batches, or mutating CPU inputs.
    """
    import numpy as np
    from torchvision import tv_tensors
    from torchvision.transforms import v2

    for key, transform in policy.transform.items():
        if key.startswith(("pixels", "goal")):
            expected = (v2.ToImage, v2.ToDtype, v2.Normalize, v2.Resize)
            if not isinstance(transform, v2.Compose) or len(transform.transforms) != 4 or any(
                    not isinstance(t, cls) for t, cls in zip(transform.transforms, expected)):
                raise ValueError("E3 requires the frozen four-stage native image pipeline")
            if transform.transforms[2].inplace:
                raise ValueError("E3 requires native non-inplace normalization")

    def prepare(_self, info_dict):
        out = {}
        for key, value in info_dict.items():
            is_numpy = isinstance(value, (np.ndarray, np.generic))
            if key in _self.process:
                if not is_numpy:
                    raise ValueError(f"Expected numpy array for key '{key}' in process, got {type(value)}")
                shape = value.shape
                if len(shape) > 2:
                    value = value.reshape(-1, *shape[2:])
                value = _self.process[key].transform(value).reshape(shape)
            if key in _self.transform:
                shape = None
                if is_numpy or torch.is_tensor(value):
                    if value.ndim > 2:
                        shape = value.shape
                        value = value.reshape(-1, *shape[2:])
                if key.startswith(("pixels", "goal")):
                    value = np.transpose(value, (0, 3, 1, 2)) if is_numpy else value.permute(0, 3, 1, 2)
                    value = _self.transform[key](tv_tensors.Image(value)).as_subclass(torch.Tensor)
                else:
                    value = torch.stack([_self.transform[key](tv_tensors.Image(x)) for x in value])
                is_numpy = isinstance(value, (np.ndarray, np.generic))
                if shape is not None:
                    value = value.reshape(*shape[:2], *value.shape[1:])
            if is_numpy and value.dtype.kind not in "USO":
                value = torch.from_numpy(value)
            out[key] = value
        return out

    had = "_prepare_info" in policy.__dict__
    saved = policy.__dict__.get("_prepare_info")
    policy._prepare_info = MethodType(prepare, policy)
    try:
        yield policy
    finally:
        if had:
            policy._prepare_info = saved
        else:
            del policy._prepare_info


@contextmanager
def deferred_refinement_checks(model):
    """Accumulate gradient finite checks on-device and reject before return.

Entry latent validation still uses the original implementation. Normalization,
zero-gradient handling and displacement constraints are unchanged. An invalid
gradient may propagate within the private loop, but no action is returned.
"""
    original_post = model.post_optimize_actions
    original_normalize = model._normalize_guidance_gradient
    had_post = "post_optimize_actions" in model.__dict__
    had_normalize = "_normalize_guidance_gradient" in model.__dict__
    saved_post = model.__dict__.get("post_optimize_actions")
    saved_normalize = model.__dict__.get("_normalize_guidance_gradient")
    finite_flags = None

    def normalize(_self, gradient, *, count_zero=True):
        nonlocal finite_flags
        if finite_flags is None:
            raise RuntimeError("deferred normalization outside refinement scope")
        finite_flags.append(torch.isfinite(gradient).all())
        reduce_dims = tuple(range(1, gradient.ndim))
        rms = gradient.float().square().mean(dim=reduce_dims, keepdim=True).sqrt()
        zero = rms <= torch.finfo(rms.dtype).eps
        normalized = gradient.float() / rms.clamp_min(torch.finfo(rms.dtype).eps)
        normalized = torch.where(zero, torch.zeros_like(normalized), normalized)
        zero_count = int(zero.sum().item()) if count_zero else 0
        return normalized.to(dtype=gradient.dtype), zero_count

    def post(_self, *args, **kwargs):
        nonlocal finite_flags
        if finite_flags is not None:
            raise RuntimeError("nested refinement is unsupported")
        finite_flags = []
        try:
            result = original_post(*args, **kwargs)
            if finite_flags and not torch.stack(finite_flags).all().item():
                raise FloatingPointError("guidance gradient contains a non-finite value")
            return result
        finally:
            finite_flags = None

    model._normalize_guidance_gradient = MethodType(normalize, model)
    model.post_optimize_actions = MethodType(post, model)
    try:
        yield model
    finally:
        if had_post:
            model.post_optimize_actions = saved_post
        else:
            del model.post_optimize_actions
        if had_normalize:
            model._normalize_guidance_gradient = saved_normalize
        else:
            del model._normalize_guidance_gradient


@contextmanager
def encoder_bf16(model):
    """Use BF16 for every encoder/projector call and return FP32 latents."""
    original = model.encode_pixels
    had_instance_method = "encode_pixels" in model.__dict__
    saved = model.__dict__.get("encode_pixels")

    def encode(_self, pixels):
        with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
            result = original(pixels)
        return result.float()

    model.encode_pixels = MethodType(encode, model)
    try:
        yield model
    finally:
        if had_instance_method:
            model.encode_pixels = saved
        else:
            del model.encode_pixels
