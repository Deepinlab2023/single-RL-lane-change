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


class ParametersPPO:
    def __init__(self):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # training loop hyperparameters
        self.num_trials = 5
        self.total_train_episodes = 1000
        self.buffer_episodes = 10  # num episodes in batch buffer
        self.t_max = 500    # max episode length
        self.opt_epochs = 10
        self.mini_batch_size = 256
        self.train_iterations = math.ceil(self.total_train_episodes / self.buffer_episodes) #top-lvl loop index
        self.test_interval = 10  # test every 10 episodes
        self.test_episodes = 10  # test 10 episodes and get average results

        # training value hyperparameters
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.99
        self.entropy_coef = 0.01
        self.eps_clip = 0.2


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