from model.actor_critic import ActorCritic
from trainer.utils import get_policy_kwargs


def get_model_class(method: str):
    method = (method or "").strip().lower()
    if method == "vanilla_ppo":
        from model.ppo_base import FCNet
        return FCNet
    if method == "diff_cvar":
        from model.diff_cvar import DiffCVaRBFQP
        return DiffCVaRBFQP
    raise ValueError(f"Unknown method {method}")


def _actor_kwargs(cfg):
    method = str(cfg.model.type)
    actor_cfg = cfg.model.get("actor", {}) or {}
    safe_dist = (
        float(actor_cfg.get("safety_margin", 0.0))
        + cfg.env.humans["radius"]
        + cfg.robot["radius"]
    )
    kwargs = {
        "robot_type": cfg.robot["type"],
        "safe_dist": safe_dist,
        "alpha": float(actor_cfg.get("alpha", 2.0)),
        "beta": float(actor_cfg.get("beta", 0.5)),
        "vmax": cfg.robot["vmax"],
        "omega_max": cfg.robot["omega_max"],
    }
    if method == "vanilla_ppo":
        kwargs.update(
            {
                "hidden_dim": int(actor_cfg.get("hidden_dim", 256)),
                "hidden_dim2": int(actor_cfg.get("hidden_dim2", 256)),
                "act": actor_cfg.get("act", "relu"),
            }
        )
    elif method == "diff_cvar":
        kwargs.update(
            {
                "hidden_dim": int(actor_cfg.get("hidden_dim", 256)),
                "control_hidden_dim": int(actor_cfg.get("control_hidden_dim", 256)),
                "scalar_hidden_dim": int(actor_cfg.get("scalar_hidden_dim", 256)),
                "act": actor_cfg.get("act", "relu"),
                "beta_min": float(actor_cfg.get("beta_min", 0.05)),
                "qp_max_iter": int(actor_cfg.get("qp_max_iter", 40)),
            }
        )

    kwargs.update(get_policy_kwargs(cfg, method))
    for cfg_key, kwarg_key in (
        ("gmm_weights", "gmm_weights"),
        ("gmm_stds", "gmm_stds"),
        ("gmm_lateral_ratio", "gmm_lateral_ratio"),
    ):
        value = actor_cfg.get(cfg_key, None)
        if value is not None:
            kwargs[kwarg_key] = value
    return kwargs


def _critic_kwargs(cfg):
    critic_cfg = cfg.model.get("critic", {}) or {}
    return {
        "hidden_dim": int(critic_cfg.get("hidden_dim", 256)),
        "hidden_dim2": int(critic_cfg.get("hidden_dim2", 256)),
        "act": critic_cfg.get("act", "relu"),
    }


def _action_std_init(cfg):
    actor_cfg = cfg.model.get("actor", {}) or {}
    return float(actor_cfg.get("action_std_init", cfg.model.get("action_std_init", 0.5)))


def build_model(config, obs_dim, act_dim, action_low=None, action_high=None):
    if action_low is None:
        action_low = [-1.0] * int(act_dim)
    if action_high is None:
        action_high = [1.0] * int(act_dim)

    method = str(config.model.type)
    policy_class = get_model_class(method)
    return ActorCritic(
        policy_class=policy_class,
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        actor_kwargs=_actor_kwargs(config),
        critic_kwargs=_critic_kwargs(config),
        action_low=action_low,
        action_high=action_high,
        action_std_init=_action_std_init(config),
    )
