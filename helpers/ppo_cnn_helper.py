import torch as th


def pre_process_cnn(obs, device):
    """
    Convert observation dictionary to tensor dictionary for CNN

    Args:
        obs: Dictionary with 'bev', 'fv', 'dynamics'
        device: torch device

    Returns:
        Dictionary of tensors ready for CNN
    """
    state = {
        'bev': th.FloatTensor(obs['bev']).unsqueeze(0).to(device) / 255.0,  # Normalize to [0, 1]
        'fv': th.FloatTensor(obs['fv']).unsqueeze(0).to(device) / 255.0,
        'dynamics': th.FloatTensor(obs['dynamics']).unsqueeze(0).to(device)
    }
    return state


class BatchProcessingCNN:
    def __init__(self):
        pass

    def collate_batch(self, buffer, device):
        """Process buffer into batch tensors for CNN-based PPO"""
        batch_states_bev, batch_states_fv, batch_states_dynamics = [], [], []
        batch_actions, batch_logp = [], []
        batch_values, batch_returns, batch_advantages = [], [], []

        for data in buffer:
            state, action, logp, value, rtrn, adv = data

            # Stack states - each state is a dict with BATCHED tensors (batch_size=1)
            # Need to concatenate along batch dimension, then squeeze
            states_bev = th.cat([s['bev'] for s in state], dim=0).to(device)  # Remove stack, use cat
            states_fv = th.cat([s['fv'] for s in state], dim=0).to(device)
            states_dynamics = th.cat([s['dynamics'] for s in state], dim=0).to(device)

            action = th.stack(action).to(device)
            logp = th.stack(logp).to(device)
            value = th.stack(value).to(device)

            batch_states_bev.append(states_bev)
            batch_states_fv.append(states_fv)
            batch_states_dynamics.append(states_dynamics)
            batch_actions.append(action)
            batch_logp.append(logp)
            batch_values.append(value)
            batch_returns.append(rtrn)
            batch_advantages.append(adv)

        # Convert to tensors
        batch_states = {
            'bev': th.cat(batch_states_bev, dim=0),
            'fv': th.cat(batch_states_fv, dim=0),
            'dynamics': th.cat(batch_states_dynamics, dim=0)
        }
        batch_actions = th.cat(batch_actions, dim=0)
        batch_logp = th.cat(batch_logp, dim=0)
        batch_values = th.cat(batch_values, dim=0).squeeze(-1)
        batch_returns = th.cat(batch_returns, dim=0)
        batch_advantages = th.cat(batch_advantages, dim=0)

        # Handle both scalar and vector advantages
        if batch_advantages.dim() > 1:
            batch_advantages = batch_advantages.squeeze(-1)

        # Normalize advantages
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        return batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages


def compute_GAE(rewards, values, gamma, gae_lambda, device):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages, returns = [], []
    R, gae = 0, 0

    for t in reversed(range(len(rewards))):
        # Compute TD error
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # Compute GAE advantage
        gae = delta + gamma * gae_lambda * gae

        # Compute discounted return
        R = rewards[t] + gamma * R

        # Store advantage and return in list
        advantages.insert(0, gae)
        returns.insert(0, R)

    # Convert lists to tensors
    returns = [th.tensor(agent_returns) for agent_returns in returns]
    returns = th.stack(returns).to(device)
    advantages = th.stack(advantages).to(device)

    del values[-1]  # Remove final next state value from buffer

    return returns, advantages