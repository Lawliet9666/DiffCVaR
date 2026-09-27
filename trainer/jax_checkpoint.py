"""JAX checkpoint helpers for the JAX diff_cvar trainer."""

from __future__ import annotations

from pathlib import Path
import json
import os
import pickle
import re
import tempfile

import jax


CHECKPOINT_FORMAT = "jax_diff_cvar_params_v1"


def _atomic_write(path, mode, writer):
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        open_kwargs = {} if "b" in mode else {"encoding": "utf-8"}
        with os.fdopen(fd, mode, **open_kwargs) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_jax_checkpoint(save_dir, params, step, performance, max_keep=10):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"ckpt_{int(step):08d}.pkl"
    payload = {
        "format": CHECKPOINT_FORMAT,
        "params": jax.device_get(params),
        "step": int(step),
    }
    _atomic_write(path, "wb", lambda handle: pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL))

    manifest_path = save_dir / "ckpt_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for ckpt_path in save_dir.glob("ckpt_*.pkl"):
        name = ckpt_path.name
        if name not in manifest:
            match = re.search(r"ckpt_(\d+)\.pkl$", name)
            manifest[name] = {
                "step": int(match.group(1)) if match else 0,
                "performance": float("-inf"),
            }
    manifest[path.name] = {"step": int(step), "performance": float(performance)}

    keep = {
        name
        for name, _info in sorted(
            manifest.items(),
            key=lambda item: item[1]["performance"],
            reverse=True,
        )[: int(max_keep)]
    }
    for name in list(manifest):
        if name not in keep:
            old_path = save_dir / name
            if old_path.exists():
                old_path.unlink()
            del manifest[name]
    _atomic_write(manifest_path, "w", lambda handle: handle.write(json.dumps(manifest, indent=2)))
    return path


def load_jax_checkpoint(path):
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported JAX checkpoint format in {path}: {payload.get('format')!r}")
    if "params" not in payload or "step" not in payload:
        raise ValueError(f"invalid JAX checkpoint payload in {path}")
    return payload
