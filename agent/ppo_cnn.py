import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal


class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from images"""

    def __init__(self, image_channels=3, image_height=224, image_width=224):
        super(CNNFeatureExtractor, self).__init__()

        # Convolutional layers for image processing (matching table architecture)
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # FIXED: stride=1
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)

        # Final flattened size: 14 × 14 × 8 = 1568
        self.linear_input_size = 14 * 14 * 8

        # Linear layer to reduce to 128 features
        self.fc = nn.Linear(self.linear_input_size, 128)

    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (batch, channels, height, width)
        Returns:
            Features of shape (batch, 128)
        """
        x = self.pool1(F.relu(self.conv1(x)))  # 224→224→112
        x = self.pool2(F.relu(self.conv2(x)))  # 112→112→56
        x = self.pool3(F.relu(self.conv3(x)))  # 56→56→28
        x = self.pool4(F.relu(self.conv4(x)))  # 28→28→14
        x = F.relu(self.conv5(x))  # 14→14
        x = F.relu(self.conv6(x))  # 14→14
        x = x.flatten(start_dim=1)  # Flatten to 1568
        x = F.relu(self.fc(x))  # Linear to 128
        return x


class PPOActorCNN(nn.Module):
    def __init__(self, image_height=224, image_width=224, dynamics_dim=7,
                 hidden_dim=256, action_dim=2, continuous=True):
        super(PPOActorCNN, self).__init__()
        self.continuous = continuous
        self.action_dim = action_dim
        self.image_height = image_height
        self.image_width = image_width

        # Two CNN extractors (one for BEV, one for FV)
        self.cnn_bev = CNNFeatureExtractor(3, image_height, image_width)
        self.cnn_fv = CNNFeatureExtractor(3, image_height, image_width)

        # Dynamics feature dimension reduction to 128
        self.pool_dynamics = nn.Linear(dynamics_dim, 128)

        # Calculate combined feature size: BEV (128) + FV (128) + Dynamics (128) = 384
        # This matches the fusion table: 1 × (128 × n) where n=3
        combined_size = 128 * 3  # 384

        # Fusion layers (matching table architecture)
        self.fc1 = nn.Linear(combined_size, 128)  # 384 → 128
        self.fc2 = nn.Linear(128, 64)  # 128 → 64
        self.fc3 = nn.Linear(64, 7)  # 64 → 7

        if continuous:
            self.fc_mean = nn.Linear(7, action_dim)
            self.fc_log_std = nn.Linear(7, action_dim)
        else:
            self.fc_action = nn.Linear(7, action_dim)

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
        # Extract features from images (each outputs 128 features)
        bev_features = self.cnn_bev(state['bev'])  # → 128
        fv_features = self.cnn_fv(state['fv'])  # → 128
        dynamics_features = F.relu(self.pool_dynamics(state['dynamics']))  # → 128

        # Concatenate all features (now 384 total)
        combined = th.cat([bev_features, fv_features, dynamics_features], dim=1)

        # Fusion layers (matching table architecture)
        x = F.relu(self.fc1(combined))  # 384 → 128
        x = F.relu(self.fc2(x))  # 128 → 64
        x = F.relu(self.fc3(x))  # 64 → 7

        if self.continuous:
            mean = self.fc_mean(x)
            log_std = self.fc_log_std(x)
            log_std = th.clamp(log_std, -20, 2)
            return th.cat([mean, log_std], dim=-1)
        else:
            logits = self.fc_action(x)
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
    def __init__(self, image_height=224, image_width=224, dynamics_dim=7, hidden_dim=256):
        super(PPOCriticCNN, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        # Two CNN extractors
        self.cnn_bev = CNNFeatureExtractor(3, image_height, image_width)
        self.cnn_fv = CNNFeatureExtractor(3, image_height, image_width)

        # Dynamics feature dimension reduction to 128
        self.pool_dynamics = nn.Linear(dynamics_dim, 128)

        # Calculate combined feature size: BEV (128) + FV (128) + Dynamics (128) = 384
        combined_size = 128 * 3  # 384

        # Fusion layers (matching table architecture)
        self.fc1 = nn.Linear(combined_size, 128)  # 384 → 128
        self.fc2 = nn.Linear(128, 64)  # 128 → 64
        self.fc3 = nn.Linear(64, 7)  # 64 → 7
        self.fc_value = nn.Linear(7, 1)  # 7 → 1 (value output)

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
        # Extract features from images (each outputs 128 features)
        bev_features = self.cnn_bev(state['bev'])  # → 128
        fv_features = self.cnn_fv(state['fv'])  # → 128
        dynamics_features = F.relu(self.pool_dynamics(state['dynamics']))  # → 128

        # Concatenate all features (now 384 total)
        combined = th.cat([bev_features, fv_features, dynamics_features], dim=1)

        # Fusion layers (matching table architecture)
        x = F.relu(self.fc1(combined))  # 384 → 128
        x = F.relu(self.fc2(x))  # 128 → 64
        x = F.relu(self.fc3(x))  # 64 → 7
        value = self.fc_value(x)  # 7 → 1
        return value

    def critic_loss(self, values, old_values, returns, eps_clip):
        value_clip = th.clamp(values, old_values - eps_clip, old_values + eps_clip)
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss