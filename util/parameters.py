import math
import torch as th

class ParametersDQN:
    def __init__(self):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # training loop hyperparameters
        self.num_trials = 5
        self.total_train_episodes = 1000
        self.batch_size = 128
        self.buffer_capacity = 10000 # number of transitions to store in buffer memory
        self.t_max = 500  # max episode length
        self.test_interval = 10  # test every 10 episodes
        self.test_episodes = 10  # test 10 episodes and get average results
        self.train_start_delay = 0 #self.batch_size#400 #min buffer length to start training

        # training value hyperparameters
        self.hidden_dim = 128
        self.actor_lr = 1e-4
        self.gamma = 0.99
        self.tau = 0.005   #update rate of the target network
        self.grad_clip = 100 #in place clipping value

        #exploration epsilon decay from 'start' to 'end' in 'decay' timesteps
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.eps_test = 0.01

        self.t_tot = 0 #timestep counter for epsilon calculation

    @property
    def epsilon(self):
        """dynamically compute epsilon based on current step as params attribute"""
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.t_tot / self.eps_decay)
        return epsilon


import torch as th
import math


class ParametersPPO:
    def __init__(self):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # --- Stability Settings ---
        self.num_trials = 1
        self.total_train_episodes = 5
        self.buffer_episodes = 2  # Collect 2 episodes before updating
        self.t_max = 100  # Max steps per episode

        # Batch Size: 32 is safe because (2 episodes * ~20 steps) > 32
        self.mini_batch_size = 32

        self.opt_epochs = 4
        self.test_interval = 2
        self.test_episodes = 2

        # --- Standard Model Params ---
        self.train_iterations = math.ceil(self.total_train_episodes / self.buffer_episodes)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.eps_clip = 0.2

        # Env specific (Trainer uses these)
        self.env_name = 'carla-cnn-v0'
        self.image_height = 64
        self.image_width = 64
        self.dynamics_dim = 7
        self.action_dim = 2
        self.continuous_actions = True

class ParametersA2C:
    def __init__(self):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # training loop hyperparameters
        self.num_trials = 5
        self.total_train_episodes = 1000
        self.batch_size = 10  # num episodes in batch buffer
        self.t_max = 500    # max episode length
        self.train_iterations = math.ceil(self.total_train_episodes / self.batch_size)  # top-lvl loop index
        self.test_interval = 10  # test every 10 episodes
        self.test_episodes = 10  # test 10 episodes and get average results

        # training value hyperparameters
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.entropy_coef = 0.005