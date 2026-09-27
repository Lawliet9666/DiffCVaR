from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)


def build_trainer(config):
    from trainer.jax_platform import configure_jax_platform_for_device

    configure_jax_platform_for_device(getattr(config, "device", "cpu"))
    from trainer.jax_ppo_base_trainer import JaxPPOBaseTrainer

    return JaxPPOBaseTrainer(config)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config"), config_name="config")
def main(config):
    assert config.run_name is not None, "run_name must be set"
    assert config.wandb_entity is not None and config.wandb_project is not None
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
