"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import glob
import json
import os
import re
import time
import wandb

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from crowd_sim.utils import absolute_obs_batch_to_relative, select_top_k_obs
from model.factory import build_model


def _safe_scalar(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.flatten()[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _set_render_safe_distance(env, actor):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if base_env is None:
        return

    safe_dist = _safe_scalar(getattr(actor, "last_r_safe", None))
    base_env.render_safe_dist = float(safe_dist) if np.isfinite(safe_dist) else None


class PPO:
    """
        PPO optimizer and rollout implementation.
    """
    def __init__(self, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Evaluation state
        self.last_eval_timestep = 0

        # Extract environment information
        self.env = env
        if hasattr(env, 'single_observation_space'):
            act_space = env.single_action_space
        else:
            act_space = env.action_space

        if getattr(self, "model_config", None) is None:
            raise ValueError("PPO requires model_config to build the train/eval model")
        self.obs_dim = int(self.model_config.model.obs_dim)
        self.act_dim = int(self.model_config.model.act_dim)
        env_act_dim = int(act_space.shape[0])
        if self.obs_dim != 6 + int(self.obs_top_k) * 6:
            raise ValueError("model.obs_dim must equal 6 + obs_top_k * 6")
        if env_act_dim != int(self.model_config.env.act_dim):
            raise ValueError("env.act_dim does not match env action space")
        if self.act_dim != env_act_dim:
            raise ValueError("model.act_dim must match env action space")

        self.model = build_model(
            self.model_config,
            self.obs_dim,
            self.act_dim,
            action_low=act_space.low,
            action_high=act_space.high,
        ).to(self.device)
        self.actor = self.model.actor
        self.critic = self.model.critic
        self.log_std = self.model.log_std
        self.act_low = self.model.act_low
        self.act_high = self.model.act_high
        self.act_scale = self.model.act_scale
        self.act_bias = self.model.act_bias

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(list(self.actor.parameters()) + [self.log_std], lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,           # timesteps so far
            'i_so_far': 0,           # iterations so far
            'batch_lens': [],        # episodic lengths in batch
            'batch_rews': [],        # episodic returns in batch
            'actor_losses': [],      # losses of actor network in current iteration
            'critic_losses': [],     # losses of critic network in current iteration
            'lr': 0,
            'actor_grads': [],       # gradients of actor network (all params)
            'critic_grads': [],      # gradients of critic network
            'u_norm_grads': [],      # gradients of policy/nominal branch
            'cbf_grads': [],         # gradients of CBF branch
            'unom_head_grads': [],   # gradients of unom head params
            'alpha_head_grads': [],  # gradients of alpha head params
            'beta_head_grads': [],   # gradients of beta head params
            'rsafe_head_grads': [],  # gradients of learned safe-radius head params
            'mu_means': [],          # mean of mu (action mean)
            'sigma_means': [],       # mean of sigma (exploration noise)
            'barrier_min_batch': [], # min barrier value in batch
            'barrier_avg_batch': [], # average barrier value in batch
            'entropy_losses': [],
            'approx_kls': [],
            'clipfracs': [],
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        num_updates = max(1, int(np.ceil(total_timesteps / max(self.timesteps_per_batch, 1))))
        start_time = time.time()
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
             
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()                     # ALG STEP 3
            batch_steps = int(np.sum(batch_lens))
            t_so_far += batch_steps

            # --- Log Min Barrier (Worst Case Safety) ---
            # We compute this outside the main training loop to avoid slowing down backprop.
            # Obs indices: 6 (rel_x), 7 (rel_y)
            with torch.no_grad():
                rel_x = batch_obs[:, 6]
                rel_y = batch_obs[:, 7]
                dist_sq = rel_x**2 + rel_y**2
                barrier = dist_sq - self.safe_dist**2
                min_barrier = torch.min(barrier).item()
                self.logger['barrier_min_batch'] = min_barrier
                self.logger['barrier_avg_batch'] = torch.mean(barrier).item()
            # ---------------------------------------------

            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(self.device)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            
            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy, mini_mean = self.evaluate(mini_obs, mini_acts)

                    # Calculate the policy ratio using log probabilities for numerical stability.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()
                    clipfrac = ((ratios - 1).abs() > self.clip).float().mean()
                    self.logger['approx_kls'].append(float(approx_kl.detach().cpu()))
                    self.logger['clipfracs'].append(float(clipfrac.detach().cpu()))

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # PPO maximizes the clipped surrogate objective; Adam minimizes its negative.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)
                    self.logger['critic_losses'].append(critic_loss.detach().cpu())

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    self.logger['entropy_losses'].append(entropy_loss.detach().cpu())
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)

                    # --- Log Gradient Norm and Distribution Stats ---
                    total_norm_sq = 0.0
                    u_norm_sq = 0.0
                    gamma_norm_sq = 0.0
                    unom_head_sq = 0.0
                    alpha_head_sq = 0.0
                    beta_head_sq = 0.0
                    rsafe_head_sq = 0.0

                    for name, p in self.actor.named_parameters():
                        if p.grad is not None:
                            g2 = p.grad.data.norm(2).item() ** 2
                            total_norm_sq += g2

                            # Policy/nominal branch gradients
                            if (
                                "unom" in name
                                or "fc31" in name
                                or "fc21" in name
                                or "fc_out" in name
                                or "fc1" in name
                                or "fc2" in name
                                or "layer" in name
                                or "robot_encoder" in name
                                or "obstacle_encoder" in name
                                or "head" in name
                            ):
                                u_norm_sq += g2

                            # CBF-like branch gradients
                            if "alpha" in name or "fc22" in name or "fc32" in name or "fc33" in name:
                                gamma_norm_sq += g2

                            # Head-specific diagnostics
                            if "fc31" in name or "unom" in name:
                                unom_head_sq += g2
                            if "fc32" in name:
                                if hasattr(self.actor, "last_alpha"):
                                    alpha_head_sq += g2
                                if hasattr(self.actor, "last_beta"):
                                    beta_head_sq += g2
                            if "fc33" in name and hasattr(self.actor, "last_r_safe"):
                                rsafe_head_sq += g2

                    # Fallback for policy families whose names don't match split.
                    if total_norm_sq > 0.0 and (u_norm_sq + gamma_norm_sq) == 0.0:
                        u_norm_sq = total_norm_sq

                    self.logger['actor_grads'].append(total_norm_sq ** 0.5)
                    self.logger['u_norm_grads'].append(u_norm_sq ** 0.5)
                    self.logger['cbf_grads'].append(gamma_norm_sq ** 0.5)
                    self.logger['unom_head_grads'].append(unom_head_sq ** 0.5)
                    self.logger['alpha_head_grads'].append(alpha_head_sq ** 0.5)
                    self.logger['beta_head_grads'].append(beta_head_sq ** 0.5)
                    self.logger['rsafe_head_grads'].append(rsafe_head_sq ** 0.5)

                    with torch.no_grad():
                        mini_mu, _ = self._squash_action(mini_mean.detach())
                        self.logger['mu_means'].append(mini_mu.mean().item())
                        self.logger['sigma_means'].append(torch.exp(self.log_std).mean().item())
                    # -----------------------------------------------

                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_norm_sq = 0.0
                    for p in self.critic.parameters():
                        if p.grad is not None:
                            critic_norm_sq += p.grad.data.norm(2).item() ** 2
                    self.logger['critic_grads'].append(critic_norm_sq ** 0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break # if kl aboves threshold
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss.detach().cpu())

            eval_metrics = None
            if self._should_eval(i_so_far, num_updates, t_so_far, total_timesteps):
                eval_metrics = self.evaluate_policy_internal(K=max(1, self.eval_episodes), step=t_so_far)
                self.save_ckpt(
                    self.save_dir,
                    step=t_so_far,
                    performance=eval_metrics["mean_return"],
                    max_keep=int(self.max_checkpoints),
                )

            sps = int(t_so_far / max(time.time() - start_time, 1e-6))
            self._log_summary(eval_metrics=eval_metrics, num_updates=num_updates, sps=sps)

    def _should_eval(self, update, num_updates, t_so_far, total_timesteps):
        if not hasattr(self, "eval_env"):
            return False
        if int(self.eval_interval) > 0:
            return update % int(self.eval_interval) == 0 or update == 1 or t_so_far >= total_timesteps
        if self.eval_freq_timesteps > 0 and (t_so_far - self.last_eval_timestep) >= self.eval_freq_timesteps:
            self.last_eval_timestep = t_so_far
            return True
        return update == num_updates


    def evaluate_policy_internal(self, K, step):
        episodes = int(K)
        ret_list = []
        len_list = []
        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_episodes = 0

        seeds = list(range(100, 1000 + 1, 100))
        for seed in tqdm(seeds, desc="Evaluating"):
            for ep in range(episodes):
                obs, _ = self.eval_env.reset(seed=seed + ep)
                done = False
                ep_ret = 0.0
                ep_len = 0
                info = {}

                while not done:
                    with torch.no_grad():
                        obs_rel = select_top_k_obs(absolute_obs_batch_to_relative(obs), self.obs_top_k)
                        obs_tensor = torch.tensor(obs_rel, dtype=torch.float).to(self.device).unsqueeze(0)
                        if hasattr(self, 'actor'):
                            policy_action = self.model.get_action_deterministic(obs_tensor)
                            action_tensor = self.model.policy_action_to_env_action(obs_tensor, policy_action)
                            action = action_tensor.detach().cpu().numpy()[0]
                        else:
                            raise ValueError("Actor model not found during evaluation")

                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = bool(terminated or truncated)
                    ep_ret += float(reward)
                    ep_len += 1

                ret_list.append(ep_ret)
                len_list.append(ep_len)
                total_episodes += 1
                success_count += int(info.get('is_success', info.get('reached', False)))
                collision_count += int(info.get('is_collision', info.get('collision', False)))
                timeout_count += int(info.get('is_timeout', info.get('timeout', False)))

        avg_ret = np.mean(ret_list)
        std_ret = np.std(ret_list)
        success_rate = success_count / total_episodes
        collision_rate = collision_count / total_episodes
        timeout_rate = timeout_count / total_episodes

        return {
            "mean_return": float(avg_ret),
            "std_return": float(std_ret),
            "success_rate": float(success_rate),
            "collision_rate": float(collision_rate),
            "timeout_rate": float(timeout_rate),
        }

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            ep_rews = np.asarray(ep_rews, dtype=np.float32).reshape(-1)
            ep_vals = np.asarray(ep_vals, dtype=np.float32).reshape(-1)
            ep_dones = np.asarray(ep_dones, dtype=np.float32).reshape(-1)

            advantages = np.zeros_like(ep_rews, dtype=np.float32)
            gae = 0.0

            # Standard GAE: done[t] indicates whether transition at step t ended the episode.
            for t in reversed(range(len(ep_rews))):
                next_value = ep_vals[t + 1] if (t + 1) < len(ep_rews) else 0.0
                next_non_terminal = 1.0 - ep_dones[t]
                delta = ep_rews[t] + self.gamma * next_value * next_non_terminal - ep_vals[t]
                gae = delta + self.gamma * self.lam * next_non_terminal * gae
                advantages[t] = gae

            batch_advantages.extend(advantages.tolist())

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return torch.tensor(batch_rtgs, dtype=torch.float)


    def rollout(self):
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        n_timeout = 0
        n_success = 0
        n_collision = 0
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            # Reset the environment. Note that obs is short for observation. 
            obs, _ = self.env.reset()
            # Initially, the game is not done
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                obs_rel = select_top_k_obs(absolute_obs_batch_to_relative(obs), self.obs_top_k)
                batch_obs.append(obs_rel)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                if self.render and len(batch_lens) == 0:
                    _set_render_safe_distance(self.env, self.actor)
                    self.env.render()

                obs_tensor = torch.tensor(obs_rel, dtype=torch.float).to(self.device)
                val = self.critic(obs_tensor)

                obs, rew, terminated, truncated, infos = self.env.step(action)
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(float(val.item()))
                # Track done for the transition at the current step (standard done[t] semantics).
                ep_dones.append(done)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    if isinstance(infos, dict):
                        n_timeout += int(infos.get('is_timeout', False))
                        n_success += int(infos.get('is_success', False))
                        n_collision += int(infos.get('is_collision', False))
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten().to(self.device)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.logger['n_timeout'] = n_timeout
        self.logger['n_success'] = n_success
        self.logger['n_collision'] = n_collision

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs_rel = select_top_k_obs(absolute_obs_batch_to_relative(obs), self.obs_top_k)
        obs = torch.tensor(obs_rel, dtype=torch.float).to(self.device)
        mean = self.actor(obs)  # latent mean (unbounded)
        dist = self._build_action_dist(mean)

        # Sample latent action then squash/mapping to real action bounds
        z = dist.rsample()
        action, u = self._squash_action(z)

        # log pi(a|s) = log p(z|s) - log|det(da/dz)|
        base_log_prob = dist.log_prob(z)
        log_prob = self._squash_log_prob(base_log_prob, u)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if self.deterministic:
            action_det, _ = self._squash_action(mean)
            return action_det.detach().cpu().numpy(), 1

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # Calculate log probabilities with squashed Gaussian change-of-variables correction.
        mean = self.actor(batch_obs)  # latent mean
        dist = self._build_action_dist(mean)
        z, u = self._unsquash_action(batch_acts)
        base_log_probs = dist.log_prob(z)
        log_probs = self._squash_log_prob(base_log_probs, u)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy(), mean

    def _build_action_dist(self, mean):
        return self.model.build_action_dist(mean)

    def _squash_action(self, z):
        return self.model.policy_action_to_env_action(None, z, return_squashed=True)

    def _unsquash_action(self, action):
        return self.model.env_action_to_policy_action(action)

    def _squash_log_prob(self, base_log_prob, u):
        return self.model.squash_log_prob(base_log_prob, u)

    def save_ckpt(self, save_dir, step, performance, max_keep=10):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"ckpt_{int(step):08d}.pt")
        torch.save({"model": self.model.state_dict(), "step": int(step)}, path)

        manifest_path = os.path.join(save_dir, "ckpt_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {}

        for ckpt_path in glob.glob(os.path.join(save_dir, "ckpt_*.pt")):
            name = os.path.basename(ckpt_path)
            if name not in manifest:
                match = re.search(r"ckpt_(\d+)\.pt$", name)
                manifest[name] = {
                    "step": int(match.group(1)) if match else 0,
                    "performance": float("-inf"),
                }
        manifest[os.path.basename(path)] = {"step": int(step), "performance": float(performance)}

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
                old_path = os.path.join(save_dir, name)
                if os.path.exists(old_path):
                    os.remove(old_path)
                del manifest[name]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return path

    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 400            # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 3e-4                                  # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.num_minibatches = 6                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0                               # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold
        self.action_std_init = 0.5                       # Initial action std (learnable)
        self.eval_interval = 100                         # Eval cadence in PPO updates
        self.eval_freq_timesteps = 200000                # Eval cadence in timesteps
        self.eval_episodes = 50                          # Episodes per periodic evaluation
        self.max_checkpoints = 10                         # Kept for config parity with diff_opt
        self.wandb_interval = 10                          # Train metric logging cadence in updates

        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.deterministic = False                      # If we're testing, don't sample actions
        self.seed = None								# Sets the seed of our program, used for reproducibility of results
        self.save_dir = './'                            # Directory to save models
        self.device = torch.device('cpu')               # Device to run on

        self.safe_dist = 0.8                            # Default safe distance for CBF
        self.alpha = 2.0                                # Default alpha for CBF
        self.beta = 0.2                                 # Default beta for CVaR
        self.robot_type = 'single_integrator'           # Default robot type
        self.vmax = 2.0                                   # Default max control input
        self.omega_max = 3.0                             # Default max angular velocity for unicycle
        self.obs_top_k = 1                               # Network input attends to first K humans (sorted nearest-first)

        for param, val in hyperparameters.items():
            if param == 'eval_env':
                self.eval_env = val
                continue
            setattr(self, param, val)

        if hasattr(self, 'save_dir'):
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Set training seed to {self.seed}", flush=True)

    def _log_summary(self, eval_metrics=None, num_updates=None, sps=0):
        t_so_far = int(self.logger['t_so_far'])
        i_so_far = int(self.logger['i_so_far'])
        lr = float(self.logger['lr'])
        avg_ep_lens = float(np.mean(self.logger['batch_lens'])) if self.logger['batch_lens'] else np.nan
        avg_ep_rews = (
            float(np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']]))
            if self.logger['batch_rews']
            else np.nan
        )
        avg_actor_loss = (
            float(np.mean([losses.float().mean().item() for losses in self.logger['actor_losses']]))
            if self.logger['actor_losses']
            else np.nan
        )
        avg_critic_loss = (
            float(np.mean([losses.float().mean().item() for losses in self.logger['critic_losses']]))
            if self.logger['critic_losses']
            else np.nan
        )
        avg_entropy = (
            float(np.mean([losses.float().mean().item() for losses in self.logger['entropy_losses']]))
            if self.logger['entropy_losses']
            else np.nan
        )
        avg_approx_kl = float(np.mean(self.logger['approx_kls'])) if self.logger['approx_kls'] else np.nan
        avg_clipfrac = float(np.mean(self.logger['clipfracs'])) if self.logger['clipfracs'] else np.nan
        avg_actor_grad = float(np.mean(self.logger['actor_grads'])) if self.logger['actor_grads'] else np.nan
        avg_critic_grad = float(np.mean(self.logger['critic_grads'])) if self.logger['critic_grads'] else np.nan
        avg_policy_grad = float(np.mean(self.logger['u_norm_grads'])) if self.logger['u_norm_grads'] else np.nan
        avg_cbf_grad = float(np.mean(self.logger['cbf_grads'])) if self.logger['cbf_grads'] else np.nan
        avg_unom_head_grad = float(np.mean(self.logger['unom_head_grads'])) if self.logger['unom_head_grads'] else np.nan
        avg_alpha_head_grad = float(np.mean(self.logger['alpha_head_grads'])) if self.logger['alpha_head_grads'] else np.nan
        avg_beta_head_grad = float(np.mean(self.logger['beta_head_grads'])) if self.logger['beta_head_grads'] else np.nan
        avg_rsafe_head_grad = float(np.mean(self.logger['rsafe_head_grads'])) if self.logger['rsafe_head_grads'] else np.nan
        avg_mu = float(np.mean(self.logger['mu_means'])) if self.logger['mu_means'] else np.nan
        avg_sigma = float(np.mean(self.logger['sigma_means'])) if self.logger['sigma_means'] else np.nan
        min_barrier = float(self.logger.get('barrier_min_batch', np.nan))
        avg_barrier = float(self.logger.get('barrier_avg_batch', np.nan))
        completed = len(self.logger['batch_lens'])

        log_dict = {
            "global_step": t_so_far,
            "charts/learning_rate": lr,
            "charts/ent_coef": float(self.ent_coef),
            "charts/sps": int(sps),
            "loss/policy": avg_actor_loss,
            "loss/value": avg_critic_loss,
            "loss/entropy": avg_entropy,
            "loss/total": avg_actor_loss + avg_critic_loss if np.isfinite(avg_actor_loss + avg_critic_loss) else np.nan,
            "diagnostics/approx_kl": avg_approx_kl,
            "diagnostics/clipfrac": avg_clipfrac,
            "diagnostics/actor_grad_norm": avg_actor_grad,
            "diagnostics/critic_grad_norm": avg_critic_grad,
            "diagnostics/policy_grad_norm": avg_policy_grad,
            "diagnostics/cbf_grad_norm": avg_cbf_grad,
            "diagnostics/unom_head_grad_norm": avg_unom_head_grad,
            "diagnostics/alpha_head_grad_norm": avg_alpha_head_grad,
            "diagnostics/beta_head_grad_norm": avg_beta_head_grad,
            "diagnostics/rsafe_head_grad_norm": avg_rsafe_head_grad,
            "diagnostics/action_mu": avg_mu,
            "diagnostics/action_sigma": avg_sigma,
            "safety/barrier_min_batch": min_barrier,
            "safety/barrier_avg_batch": avg_barrier,
        }
        if completed > 0:
            log_dict["train/episodic_return"] = avg_ep_rews
            log_dict["train/episodic_length"] = avg_ep_lens
            log_dict["train/success_rate"] = float(self.logger.get('n_success', 0)) / completed
            log_dict["train/collision_rate"] = float(self.logger.get('n_collision', 0)) / completed
            log_dict["train/timeout_rate"] = float(self.logger.get('n_timeout', 0)) / completed
        if eval_metrics is not None:
            log_dict["charts/eval_return_mean"] = eval_metrics["mean_return"]
            log_dict["charts/eval_return_std"] = eval_metrics["std_return"]
            log_dict["charts/eval_success_rate"] = eval_metrics["success_rate"]
            log_dict["charts/eval_collision_rate"] = eval_metrics["collision_rate"]
            log_dict["charts/eval_timeout_rate"] = eval_metrics["timeout_rate"]

        if wandb.run is not None:
            wandb_log = {key: value for key, value in log_dict.items() if not key.startswith("train/")}
            if int(self.wandb_interval) <= 0:
                raise ValueError("wandb_interval must be > 0")
            if completed > 0 and i_so_far % int(self.wandb_interval) == 0:
                wandb_log.update({key: value for key, value in log_dict.items() if key.startswith("train/")})
            wandb.log(wandb_log, step=t_so_far)

        eval_return = eval_metrics["mean_return"] if eval_metrics is not None else float("nan")
        eval_success = eval_metrics["success_rate"] if eval_metrics is not None else float("nan")
        total_updates = int(num_updates or i_so_far)
        print(
            f"update {i_so_far:4d}/{total_updates} | steps {t_so_far:8d} | "
            f"eval {eval_return:8.2f} | success {eval_success:.3f} | sps {int(sps)}",
            flush=True,
        )

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['actor_grads'] = []
        self.logger['critic_grads'] = []
        self.logger['u_norm_grads'] = []
        self.logger['cbf_grads'] = []
        self.logger['unom_head_grads'] = []
        self.logger['alpha_head_grads'] = []
        self.logger['beta_head_grads'] = []
        self.logger['rsafe_head_grads'] = []
        self.logger['mu_means'] = []
        self.logger['sigma_means'] = []
        self.logger['barrier_min_batch'] = []
        self.logger['barrier_avg_batch'] = []
        self.logger['entropy_losses'] = []
        self.logger['approx_kls'] = []
        self.logger['clipfracs'] = []
        self.logger['n_timeout'] = 0
        self.logger['n_success'] = 0
        self.logger['n_collision'] = 0
