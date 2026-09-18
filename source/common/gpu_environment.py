"""Keep CUDA compute and MuJoCo EGL rendering on one permitted GPU."""

from __future__ import annotations

import os


PERMITTED_PHYSICAL_GPUS = frozenset(range(8))


def configure_mujoco_egl_device() -> int:
    """Bind dm_control's EGL device before importing any rendering modules."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise EnvironmentError(
            "CUDA_VISIBLE_DEVICES must explicitly select exactly one GPU0-7"
        )
    try:
        selected = tuple(int(token.strip()) for token in visible.split(","))
    except ValueError as exc:
        raise EnvironmentError(
            "CUDA_VISIBLE_DEVICES must contain exactly one integer GPU ID"
        ) from exc
    if len(selected) != 1:
        raise EnvironmentError(
            "MuJoCo EGL jobs must select exactly one physical GPU"
        )
    physical_gpu = selected[0]
    if physical_gpu not in PERMITTED_PHYSICAL_GPUS:
        raise EnvironmentError(
            f"prohibited CUDA_VISIBLE_DEVICES={visible!r}; only GPU0-7 are allowed"
        )

    configured = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if configured is not None and configured != str(physical_gpu):
        raise EnvironmentError(
            "MUJOCO_EGL_DEVICE_ID differs from CUDA_VISIBLE_DEVICES: "
            f"{configured!r} != {physical_gpu!r}"
        )
    platform = os.environ.get("PYOPENGL_PLATFORM")
    if platform is not None and platform != "egl":
        raise EnvironmentError(
            f"PYOPENGL_PLATFORM must be 'egl', got {platform!r}"
        )
    backend = os.environ.get("MUJOCO_GL")
    if backend is not None and backend != "egl":
        raise EnvironmentError(f"MUJOCO_GL must be 'egl', got {backend!r}")

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical_gpu)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
    return physical_gpu


__all__ = ["PERMITTED_PHYSICAL_GPUS", "configure_mujoco_egl_device"]
