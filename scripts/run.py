"""Hydra entrypoint for PPO training."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.ppo_trainer import PPOTrainer
from scripts.override_validation import reject_append_delete_overrides


OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config"), config_name="config")
def main(cfg: DictConfig):
    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    try:
        reject_append_delete_overrides(sys.argv[1:], context="training")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2)
    main()
