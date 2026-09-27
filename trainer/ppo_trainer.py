"""PPO training framework wrapper."""

import os
import torch

from model.factory import get_model_class
from trainer.trainer import Trainer
from trainer.vec_ppo import VecPPO


class PPOTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_env()
        self.model = self.build_model()

    def build_model(self):
        num_envs = int(self.cfg.trainer.num_envs)
        method = str(self.cfg.model.type)
        print(f"Training with {num_envs} vectorized environments", flush=True)
        policy_class = get_model_class(method)
        print(f"Algorithm: {method}, Policy: {policy_class.__name__}", flush=True)

        seeds = [int(self.hyperparameters["seed"]) + i for i in range(num_envs)]
        self.train_envs.reset(seed=seeds)
        model = VecPPO(
            env=self.train_envs,
            num_envs=num_envs,
            **self.hyperparameters,
        )
        self.load_warm_start(model)
        return model

    def load_warm_start(self, model):
        loaded_parts = []
        checkpoint = str(self.cfg.get("checkpoint", "") or "").strip()

        if checkpoint:
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if not (isinstance(state, dict) and "model" in state):
                raise ValueError(f"Expected checkpoint with a top-level 'model' key: {checkpoint}")
            model.model.load_state_dict(state["model"], strict=True)
            loaded_parts.append(f"model={checkpoint}")

        if loaded_parts:
            print(f"Warm start loaded: {', '.join(loaded_parts)}", flush=True)
        else:
            print("Training from scratch.", flush=True)

    def train(self):
        try:
            self.model.learn(total_timesteps=int(self.cfg.trainer.total_timesteps))
        finally:
            self.close()
