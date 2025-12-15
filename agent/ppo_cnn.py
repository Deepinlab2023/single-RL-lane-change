import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal


class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from images"""

    def __init__(self, image_channels=3, image_height=64, image_width=64):
        super(CNNFeatureExtractor, self).__init__()

        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_height, 8, 4), 4, 2), 3, 1)
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_width, 8, 4), 4, 2), 3, 1)

        self.linear_input_size = conv_h * conv_w * 64

    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (batch, channels, height, width)
        Returns:
            Flattened features
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        return x


class PPOActorCNN(nn.Module):
    def __init__(self, image_height=64, image_width=64, dynamics_dim=7,
                 hidden_dim=256, action_dim=2, continuous=True):
        super(PPOActorCNN, self).__init__()
        self.continuous = continuous
        self.action_dim = action_dim
        self.image_height = image_height
        self.image_width = image_width

        # Two CNN extractors (one for BEV, one for FV)
        self.cnn_bev = CNNFeatureExtractor(3, image_height, image_width)
        self.cnn_fv = CNNFeatureExtractor(3, image_height, image_width)

        # Feature dimension reduction (matching table architecture)
        # Reduce CNN outputs to 128 each before fusion
        self.pool_bev = nn.Linear(self.cnn_bev.linear_input_size, 128)
        self.pool_fv = nn.Linear(self.cnn_fv.linear_input_size, 128)
        self.pool_dynamics = nn.Linear(dynamics_dim, 128)

        # Calculate combined feature size after pooling
        combined_size = 128 + 128 + 128  # BEV + FV + Dynamics = 384

        # Fully connected layers
        self.fc1 = nn.Linear(combined_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if continuous:
            self.fc_mean = nn.Linear(hidden_dim, action_dim)
            self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """
        Args:
            state: Dictionary with keys 'bev', 'fv', 'dynamics'
                - bev: (batch, 3, H, W)
                - fv: (batch, 3, H, W)
                - dynamics: (batch, dynamics_dim)
        """
        # Extract features from images
        bev_features = self.cnn_bev(state['bev'])
        fv_features = self.cnn_fv(state['fv'])

        # Reduce dimensions (matching table architecture)
        bev_pooled = F.relu(self.pool_bev(bev_features))
        fv_pooled = F.relu(self.pool_fv(fv_features))
        dynamics_pooled = F.relu(self.pool_dynamics(state['dynamics']))

        # Concatenate all features (now 384 total)
        combined = th.cat([bev_pooled, fv_pooled, dynamics_pooled], dim=1)

        # Process through FC layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        if self.continuous:
            mean = self.fc_mean(x)
            log_std = self.fc_log_std(x)
            log_std = th.clamp(log_std, -20, 2)
            return th.cat([mean, log_std], dim=-1)
        else:
            logits = self.fc3(x)
            return logits

    def select_action(self, output):
        if self.continuous:
            mean, log_std = th.chunk(output, 2, dim=-1)
            std = log_std.exp()
            action_dist = Normal(mean, std)
            action = action_dist.sample()
            logp = action_dist.log_prob(action).sum(dim=-1)
            return action, logp, action_dist
        else:
            action_dist = Categorical(logits=output)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            return action, logp, action_dist

    def actor_loss(self, logp, old_logp, advantages, eps_clip):
        imp_weights = th.exp(logp - old_logp)
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        loss = -th.min(surr1, surr2).mean()
        return loss


class PPOCriticCNN(nn.Module):
    def __init__(self, image_height=64, image_width=64, dynamics_dim=7, hidden_dim=256):
        super(PPOCriticCNN, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        # Two CNN extractors
        self.cnn_bev = CNNFeatureExtractor(3, image_height, image_width)
        self.cnn_fv = CNNFeatureExtractor(3, image_height, image_width)

        # Feature dimension reduction (matching table architecture)
        self.pool_bev = nn.Linear(self.cnn_bev.linear_input_size, 128)
        self.pool_fv = nn.Linear(self.cnn_fv.linear_input_size, 128)
        self.pool_dynamics = nn.Linear(dynamics_dim, 128)

        # Calculate combined feature size after pooling
        combined_size = 128 + 128 + 128  # BEV + FV + Dynamics = 384

        # Fully connected layers
        self.fc1 = nn.Linear(combined_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """
        Args:
            state: Dictionary with keys 'bev', 'fv', 'dynamics'
        """
        # Extract features from images
        bev_features = self.cnn_bev(state['bev'])
        fv_features = self.cnn_fv(state['fv'])

        # Reduce dimensions (matching table architecture)
        bev_pooled = F.relu(self.pool_bev(bev_features))
        fv_pooled = F.relu(self.pool_fv(fv_features))
        dynamics_pooled = F.relu(self.pool_dynamics(state['dynamics']))

        # Concatenate all features (now 384 total)
        combined = th.cat([bev_pooled, fv_pooled, dynamics_pooled], dim=1)

        # Process through FC layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

    def critic_loss(self, values, old_values, returns, eps_clip):
        value_clip = th.clamp(values, old_values - eps_clip, old_values + eps_clip)
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss