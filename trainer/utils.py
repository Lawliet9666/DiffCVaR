"""Shared runtime helpers for training and evaluation entrypoints."""

import random

import numpy as np
import torch


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested, default="cuda"):
    requested = str(requested or "auto").lower()
    if requested == "auto":
        requested = default

    if requested == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
            return device
        print("GPU requested but not available. Using CPU.", flush=True)

    print("Using CPU.", flush=True)
    return torch.device("cpu")


def get_policy_kwargs(config, method):
    method = str(method or "").strip().lower()
    if method != "diff_cvar":
        return {}

    actor_cfg = config.model.get("actor", {}) or {}
    gmm_cfg = dict(config.env.humans.get("gmm", {}))
    return {
        "gmm_weights": actor_cfg.get("gmm_weights", gmm_cfg.get("weights")),
        "gmm_stds": actor_cfg.get("gmm_stds", gmm_cfg.get("stds")),
        "gmm_lateral_ratio": actor_cfg.get(
            "gmm_lateral_ratio", gmm_cfg.get("lateral_ratio", 0.3)
        ),
    }
