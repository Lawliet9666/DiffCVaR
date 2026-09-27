"""CrowdSim package initialization.

Registers Gymnasium environments for convenience when using gym.make().
"""

from gymnasium.envs.registration import register, registry


def _safe_register(env_id: str, entry_point: str) -> None:
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)


_safe_register("CrowdSim-v0", "crowd_sim.env.social_nav:SocialNav")
_safe_register("CrowdSimVarNum-v0", "crowd_sim.env.social_nav_var_num:SocialNavVarNum")
