"""JAX-only model package and lazy policy selection."""

from importlib import import_module


_JAX_MODEL_MODULES = {
    "diff_cvar_gnn": "model.jax_diff_cvar_gnn",
    "diff_cvar_mlp": "model.jax_diff_cvar_mlp",
    "ppo_gnn": "model.jax_ppo_gnn",
    "ppo_mlp": "model.jax_ppo_mlp",
}


def get_jax_model_module(model_cfg):
    model_type = str(model_cfg.get("type") if isinstance(model_cfg, dict) else model_cfg.type)
    module_name = _JAX_MODEL_MODULES.get(model_type)
    if module_name is None:
        raise ValueError(f"unsupported model.type={model_type!r}; expected one of {tuple(_JAX_MODEL_MODULES)}")
    return import_module(module_name)


__all__ = ["get_jax_model_module"]
