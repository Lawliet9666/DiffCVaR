import torch
import numpy as np
from crowd_sim.utils import absolute_obs_batch_to_relative, select_top_k_obs
from trainer.ppo import PPO

class VecPPO(PPO):
    def __init__(self, env, num_envs, **hyperparameters):
        super().__init__(env, **hyperparameters)
        self.num_envs = num_envs

    def rollout(self):
        """
            Rollout logic for Vectorized Environments.
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Buffers for each environment
        env_obs = [[] for _ in range(self.num_envs)]
        env_acts = [[] for _ in range(self.num_envs)]
        env_log_probs = [[] for _ in range(self.num_envs)]
        env_rews = [[] for _ in range(self.num_envs)]
        env_dones = [[] for _ in range(self.num_envs)]

        n_timeout = 0
        n_success = 0
        n_collision = 0

        # Reset all environments
        obs, _ = self.env.reset()
        obs = absolute_obs_batch_to_relative(obs)
        obs = select_top_k_obs(obs, self.obs_top_k)

        t_so_far = 0
        
        # We continue until we have collected enough timesteps in COMPLETED episodes
        while t_so_far < self.timesteps_per_batch:
            # Get actions for all envs
            # obs is (num_envs, obs_dim) which works with FeedForwardNN
            actions, log_probs = self.get_action(obs)
            
            # Step the vectorized environment
            next_obs, rews, terminations, truncations, infos = self.env.step(actions)
            next_obs = absolute_obs_batch_to_relative(next_obs)
            next_obs = select_top_k_obs(next_obs, self.obs_top_k)
            
            dones = terminations | truncations

            for i in range(self.num_envs):
                # Store step data
                env_obs[i].append(obs[i])
                env_acts[i].append(actions[i])
                env_log_probs[i].append(log_probs[i])
                env_rews[i].append(rews[i])
                env_dones[i].append(bool(dones[i]))
                
                if dones[i]:
                    # Standardize info access (Gymnasium tuple-of-dicts vs Legacy dict-of-arrays)
                    info = infos[i] if isinstance(infos, (tuple, list)) else {k: v[i] for k, v in infos.items() if v is not None}
                    
                    # Check 'final_info' for meaningful terminal state (Gymnasium auto-reset)
                    info = info.get('final_info') or info

                    n_timeout += int(info.get('is_timeout', False))
                    n_success += int(info.get('is_success', False))
                    n_collision += int(info.get('is_collision', False))

                    # Episode finished for env i
                    ep_len = len(env_rews[i])
                    ep_rews = env_rews[i]
                    
                    # Store episode data to batch
                    batch_obs.extend(env_obs[i])
                    batch_acts.extend(env_acts[i])
                    batch_log_probs.extend(env_log_probs[i])
                    batch_rews.append(ep_rews)
                    batch_lens.append(ep_len)

                    # Compute value estimates for this episode
                    with torch.no_grad():
                        ep_obs_tensor = torch.tensor(np.array(env_obs[i]), dtype=torch.float).to(self.device)
                        ep_vals = self.critic(ep_obs_tensor).squeeze().detach().cpu().numpy().tolist()
                    if not isinstance(ep_vals, list):
                        ep_vals = [ep_vals]
                    batch_vals.append(ep_vals)
                    batch_dones.append(env_dones[i])
                    
                    # RTGs are computed after collecting complete episodes.
                    
                    t_so_far += ep_len
                    
                    # Reset buffers for env i
                    env_obs[i] = []
                    env_acts[i] = []
                    env_log_probs[i] = []
                    env_rews[i] = []
                    env_dones[i] = []
            
            # Update obs
            obs = next_obs

        # Convert to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(self.device)
        
        # Log
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.logger['n_timeout'] = n_timeout
        self.logger['n_success'] = n_success
        self.logger['n_collision'] = n_collision

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
