import sys
from copy import deepcopy
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
                    next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

                    # Store transition
                    state_history.append(state)
                    action_history.append(action)
                    logp_history.append(logp)
                    reward_history.append(reward)
                    value_history.append(value)

                    total_reward += reward
                    obs = next_obs

                    # Logic for episode termination/truncation - ALWAYS append next value
                    if truncated:
                        # Episode truncated (time limit) - bootstrap from next state
                        next_state = pre_process_cnn(obs, device)
                        with th.no_grad():
                            next_value = critic(next_state)
                        value_history.append(next_value)
                        break

                    if terminated:
                        # Episode terminated (collision/route complete) - next value is 0
                        next_value = th.zeros_like(value)
                        value_history.append(next_value)
                        break

                # IMPORTANT: If we exit the loop without break, we still need to append next value
                # This happens when t == params.t_max - 1
                if len(value_history) == len(reward_history):
                    # No termination or truncation happened, bootstrap from final state
                    next_state = pre_process_cnn(obs, device)
                    with th.no_grad():
                        next_value = critic(next_state)
                    value_history.append(next_value)

                episode_rewards.append(total_reward)
                n_ep += 1

                # Compute returns and advantages for episode, add episode to buffer
                returns, advantages = compute_GAE(reward_history, value_history, params.gamma, params.gae_lambda,
                                                  device)
                buffer.append((state_history, action_history, logp_history, value_history, returns, advantages))

                # Test at interval and print result
                if n_ep % params.test_interval == 0:
                    print(f'Episode {n_ep}: Training reward: {total_reward:.2f}')

            # Process buffer once full
            batch_process = BatchProcessingCNN()
            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages = batch_process.collate_batch(
                buffer, params.device)

            # Create custom dataset that handles dict states
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
            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.mini_batch_size, shuffle=True)

            # Optimization loop
            for _ in range(params.opt_epochs):
                for batch in dataloader:
                    states_mb, actions_mb, logp_mb, values_mb, returns_mb, advantages_mb = batch

                    # Move to device
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