import sys
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch as th
from torch.distributions import Normal

from agent.ppo_cnn import PPOActorCNN, PPOCriticCNN
from helpers.ppo_cnn_helper import BatchProcessingCNN, compute_GAE, pre_process_cnn


class PPOCNNtrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device

        actor = PPOActorCNN(
            image_height=params.image_height,
            image_width=params.image_width,
            dynamics_dim=params.dynamics_dim,
            hidden_dim=params.actor_hidden_dim,
            action_dim=params.action_dim,
            continuous=True
        ).to(device)

        critic = PPOCriticCNN(
            image_height=params.image_height,
            image_width=params.image_width,
            dynamics_dim=params.dynamics_dim,
            hidden_dim=params.critic_hidden_dim
        ).to(device)

        actor_opt = th.optim.Adam(actor.parameters(), lr=params.actor_lr)
        critic_opt = th.optim.Adam(critic.parameters(), lr=params.critic_lr)

        episode_rewards = []
        test_rewards = []
        tests_info = []

        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []

            for ep in range(params.buffer_episodes):
                state_history = []
                action_history = []
                logp_history = []
                reward_history = []
                value_history = []

                obs, _ = env.reset()
                total_reward = 0

                for t in range(params.t_max):
                    state = pre_process_cnn(obs, device)

                    # Select action, compute value estimate
                    with th.no_grad():
                        output = actor(state)
                        action, logp, _ = actor.select_action(output)
                        value = critic(state)

                    # Take a step
                    next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

                    # Store transition
                    state_history.append(state)
                    action_history.append(action)
                    logp_history.append(logp)
                    reward_history.append(reward)
                    value_history.append(value)

                    total_reward += reward
                    obs = next_obs

                    # Logic for episode termination/truncation
                    if truncated:
                        next_state = pre_process_cnn(obs, device)
                        with th.no_grad():
                            next_value = critic(next_state)
                        value_history.append(next_value)
                        break

                    if terminated:
                        next_value = th.zeros_like(value)
                        value_history.append(next_value)
                        break

                # Handle timeout without truncated flag
                if len(value_history) == len(reward_history):
                    next_state = pre_process_cnn(obs, device)
                    with th.no_grad():
                        next_value = critic(next_state)
                    value_history.append(next_value)

                episode_rewards.append(total_reward)
                n_ep += 1

                # Compute returns and advantages
                returns, advantages = compute_GAE(reward_history, value_history, params.gamma, params.gae_lambda,
                                                  device)
                buffer.append((state_history, action_history, logp_history, value_history, returns, advantages))

                # Test at interval
                if n_ep % params.test_interval == 0:
                    test_reward, test_info = self.test(deepcopy(actor), params)
                    test_rewards.append(test_reward)
                    tests_info.append(test_info)

                    print(f'Episode {n_ep}: Training reward: {total_reward:.2f}, '
                          f'Test reward: {test_reward:.2f}, '
                          f'Test info: dist={test_info.get("avg_distance", 0):.2f}m, '
                          f'speed={test_info.get("avg_speed", 0):.2f}m/s, '
                          f'deviation={test_info.get("avg_deviation", 0):.2f}m, '
                          f'collision={test_info.get("collision_rate", 0):.1f}%')

            # Process buffer
            batch_process = BatchProcessingCNN()
            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages = batch_process.collate_batch(
                buffer, params.device)

            class DictStateDataset(th.utils.data.Dataset):
                def __init__(self, states, actions, logp, values, returns, advantages):
                    self.states = states
                    self.actions = actions
                    self.logp = logp
                    self.values = values
                    self.returns = returns
                    self.advantages = advantages
                    self.length = actions.shape[0]

                def __len__(self):
                    return self.length

                def __getitem__(self, idx):
                    return (
                        {'bev': self.states['bev'][idx], 'fv': self.states['fv'][idx],
                         'dynamics': self.states['dynamics'][idx]},
                        self.actions[idx],
                        self.logp[idx],
                        self.values[idx],
                        self.returns[idx],
                        self.advantages[idx]
                    )

            dataset = DictStateDataset(batch_states, batch_actions, batch_logp, batch_values, batch_returns,
                                       batch_advantages)

            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.mini_batch_size, shuffle=True,
                                                  drop_last=False)

            # Update loop
            for _ in range(params.opt_epochs):
                for batch in dataloader:
                    states_mb, actions_mb, logp_mb, values_mb, returns_mb, advantages_mb = batch

                    states_mb = {
                        'bev': states_mb['bev'].to(params.device),
                        'fv': states_mb['fv'].to(params.device),
                        'dynamics': states_mb['dynamics'].to(params.device)
                    }
                    actions_mb = actions_mb.to(params.device)
                    logp_mb = logp_mb.to(params.device)
                    values_mb = values_mb.to(params.device)
                    returns_mb = returns_mb.to(params.device)
                    advantages_mb = advantages_mb.to(params.device)

                    # Critic Update
                    critic_opt.zero_grad()
                    values_new = critic(states_mb)
                    critic_loss = critic.critic_loss(values_new, values_mb, returns_mb, params.eps_clip)
                    critic_loss.backward()
                    critic_opt.step()

                    # Actor Update
                    actor_opt.zero_grad()
                    actor_output = actor(states_mb)
                    mean, log_std = th.chunk(actor_output, 2, dim=-1)
                    std = log_std.exp()
                    dist = Normal(mean, std)
                    logp_new = dist.log_prob(actions_mb).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    actor_loss = actor.actor_loss(logp_new, logp_mb, advantages_mb, params.eps_clip)
                    actor_loss = actor_loss - params.entropy_coef * entropy
                    actor_loss.backward()
                    actor_opt.step()

        print("Training done")
        return episode_rewards, test_rewards, tests_info

    @staticmethod
    def test(actor, params):
        """Test agent and collect comprehensive metrics."""
        test_env = gym.make(params.env_name)

        all_episode_info = []
        test_rewards = []

        for i in range(params.test_episodes):
            total_reward = 0
            obs, _ = test_env.reset()
            info = {}

            for t in range(params.t_max):
                state = pre_process_cnn(obs, params.device)

                with th.no_grad():
                    output = actor(state)
                    action, _, _ = actor.select_action(output)

                next_obs, reward, done, trunc, info = test_env.step(action.cpu().numpy())

                total_reward += reward
                obs = next_obs

                if done or trunc:
                    break

            all_episode_info.append(info)
            test_rewards.append(total_reward)

        test_env.close()

        if not all_episode_info:
            return 0.0, {}

        # Safe helper to average metrics across episodes
        def get_avg(key):
            vals = [i.get(key, 0) for i in all_episode_info]
            return np.mean(vals) if vals else 0

        def get_std(key):
            vals = [i.get(key, 0) for i in all_episode_info]
            return np.std(vals) if vals else 0

        # --- COMPLETE METRIC LIST (Fixes all KeyErrors) ---
        aggregated_info = {
            # Core
            'avg_distance': get_avg('distance_travelled_m'),
            'std_distance': get_std('distance_travelled_m'),

            'avg_speed': get_avg('avg_speed_m_s'),
            'std_speed': get_std('avg_speed_m_s'),

            'avg_deviation': get_avg('avg_deviation_m'),
            'std_deviation': get_std('avg_deviation_m'),

            # Benchmarker Specific Keys
            'avg_angle': get_avg('avg_angle_deg'),  # Mapped from 'avg_angle_deg'
            'avg_steering_change': get_avg('avg_steering_change'),
            'avg_lateral_accel': get_avg('avg_lateral_accel'),

            # Rates
            'collision_rate': sum(i.get('collision', False) for i in all_episode_info) / len(all_episode_info) * 100,
            'success_rate': sum(i.get('termination_reason', '') == 'ROUTE_COMPLETED' for i in all_episode_info) / len(
                all_episode_info) * 100,

            # Full history for deeper analysis if needed
            'all_episodes': all_episode_info
        }

        return np.mean(test_rewards), aggregated_info