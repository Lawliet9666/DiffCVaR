"""JAX platform selection helpers for the JAX PPO path."""

from __future__ import annotations

import os


def _normalize_device(device) -> str:
    return str(device).strip().lower()


def jax_platform_from_device(device) -> str:
    """Map the user-facing device config to a JAX platform name."""
    value = _normalize_device(device)
    if value in {"auto", ""}:
        return "auto"
    if value == "cpu":
        return "cpu"
    if value.startswith("cuda") or value.startswith("gpu"):
        return "cuda"
    raise ValueError(f"JAX training supports device=auto, device=cpu, or device=cuda*, got {device!r}")


def _platform_aliases(platform: str) -> set[str]:
    if platform == "cuda":
        return {"cuda", "gpu"}
    return {platform}


def _set_default_env(name: str, value: str):
    if not os.environ.get(name):
        os.environ[name] = value


def _append_xla_flag_if_missing(flag: str):
    existing = os.environ.get("XLA_FLAGS", "")
    flags = existing.split()
    if flag not in flags:
        os.environ["XLA_FLAGS"] = " ".join([*flags, flag])


def configure_jax_platform_for_device(device) -> str:
    """Set JAX_PLATFORMS before JAX is imported.

    device=auto leaves platform choice to JAX, so CUDA is used when JAX can see
    it and CPU is used otherwise. Explicit device=cpu/cuda remains strict. If
    the environment already pins a different first JAX platform for an explicit
    device, fail instead of letting JAX silently use a different backend.
    """
    requested = jax_platform_from_device(device)
    _set_default_env("JAX_ENABLE_X64", "true")
    existing = os.environ.get("JAX_PLATFORMS")

    if requested == "auto":
        first = existing.split(",")[0].strip().lower() if existing else "auto"
        if not existing or first in _platform_aliases("cuda"):
            _set_default_env("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            _append_xla_flag_if_missing("--xla_gpu_autotune_level=0")
        return "cuda" if first in _platform_aliases("cuda") else first
    if existing:
        first = existing.split(",")[0].strip().lower()
        if first not in _platform_aliases(requested):
            raise RuntimeError(
                f"device={device!r} requests JAX platform {requested!r}, "
                f"but JAX_PLATFORMS={existing!r} is already set"
            )
    else:
        os.environ["JAX_PLATFORMS"] = requested

    if requested == "cuda":
        _set_default_env("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        _append_xla_flag_if_missing("--xla_gpu_autotune_level=0")
    return requested


def validate_jax_backend_for_device(jax_module, device) -> str:
    """Validate that JAX initialized the backend requested by device."""
    requested = jax_platform_from_device(device)
    backend = str(jax_module.default_backend()).lower()
    if requested == "auto":
        return backend
    if backend not in _platform_aliases(requested):
        raise RuntimeError(
            f"device={device!r} requires JAX backend {requested!r}, "
            f"but jax.default_backend() is {backend!r}"
        )
    if requested == "cuda":
        devices = []
        for platform in ("gpu", "cuda"):
            try:
                devices = list(jax_module.devices(platform))
            except RuntimeError:
                devices = []
            if devices:
                break
        if not devices:
            raise RuntimeError("device=cuda was requested, but JAX reports no GPU devices")
    return backend


def jax_device_runtime_info(jax_module) -> dict:
    """Return JAX logical devices plus CUDA_VISIBLE_DEVICES mapping."""
    devices = [str(device) for device in jax_module.devices()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_ids = [item.strip() for item in visible.split(",") if item.strip()]
    mapping = {}
    for device in devices:
        logical_id = device.rsplit(":", 1)[-1]
        if logical_id.isdigit() and int(logical_id) < len(physical_ids):
            mapping[device] = physical_ids[int(logical_id)]
    return {
        "runtime_jax_devices": devices,
        "runtime_cuda_visible_devices": visible,
        "runtime_cuda_logical_to_physical": mapping,
    }


def format_jax_device_runtime_info(info: dict) -> str:
    parts = []
    if visible := info.get("runtime_cuda_visible_devices"):
        parts.append(f"CUDA_VISIBLE_DEVICES={visible}")
    parts.extend(
        f"{logical}->physical_gpu{physical}"
        for logical, physical in (info.get("runtime_cuda_logical_to_physical") or {}).items()
    )
    suffix = "; ".join(parts)
    return f"; {suffix}" if suffix else ""
