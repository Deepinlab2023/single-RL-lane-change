import argparse
import time
import gymnasium as gym
from gymnasium.envs.registration import register  # <--- Added Import
from envs.sumo_env import SumoEnv
from envs.carla_cnn_env import CarlaCNNEnv

from runner.runner import ALGOrunner
from train.ppo_trainer import PPOtrainer
from train.ppo_cnn_trainer import PPOCNNtrainer
from train.dqn_trainer import DQNtrainer
from train.a2c_trainer import A2Ctrainer
from util.parameters import ParametersPPO, ParametersDQN, ParametersA2C


def main():
    parser = argparse.ArgumentParser(
        description="Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True,
                        help='Choose: "cartpole", "pong", "sumo", or "carla".')
    parser.add_argument('--algo', type=str, required=True,
                        help='Choose: "dqn", "ppo", or "a2c".')
    parser.add_argument('--town', type=str, default='Town04',
                        help='CARLA town to use: "Town01", "Town02", "Town03", "Town04", "Town05", etc. (default: Town04)')
    parser.add_argument('--render-freq', type=int, default=1,
                        help='Render every N steps (default: 1 = every step)')
    args = parser.parse_args()

    # --- Select environment ---
    if args.env.lower() == 'cartpole':
        env_name = 'CartPole-v1'
        env = gym.make(env_name)

    elif args.env.lower() == 'pong':
        raise ValueError("Pong environment not implemented.")

    elif args.env.lower() == 'sumo':
        env_name = 'sumo-v0'
        # Sumo usually needs manual instantiation or complex registration due to the object return
        # If your SumoEnv works with make(), register it. Otherwise, keep as is.
        env = SumoEnv(
            max_steps=40,
            v_max=25,
            render_mode=None,
            step_length=0.2,
            decision_steps=5,
            w_collision=0.1,
            w_efficiency=0.1,
            w_lane_change=0.1,
            w_completion=0.5
        )

    elif args.env.lower() == 'carla':
        env_name = 'carla-cnn-v0'

        # --- FIX: Register the environment so the Trainer can call gym.make() ---
        # We check if it's already registered to avoid double-registration errors
        if env_name not in gym.envs.registry:
            register(
                id=env_name,
                entry_point='envs.carla_cnn_env:CarlaCNNEnv',  # Ensure this path matches your folder structure
                kwargs={
                    'host': 'localhost',
                    'port': 2000,
                    'num_surrounding_vehicles': 30,
                    'max_steps': 1000,
                    'town': args.town  # Pass town parameter from command line
                }
            )

        # Now create the env using gym.make so it's consistent with the trainer's logic
        env = gym.make(env_name)
        print(f"CARLA Environment created with {args.town}")

    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # --- Select algorithm ---
    if args.algo.lower() == 'ppo':
        params = ParametersPPO()

        # Use CNN trainer for CARLA
        if args.env.lower() == 'carla':
            trainer = PPOCNNtrainer
        else:
            trainer = PPOtrainer

    elif args.algo.lower() == 'a2c':
        params = ParametersA2C()
        trainer = A2Ctrainer
    elif args.algo.lower() == 'dqn':
        params = ParametersDQN()
        trainer = DQNtrainer
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # --- Set parameters ---
    params.env_name = env_name

    # Handle different observation space types
    if args.env.lower() == 'carla':
        # CARLA with CNN - set image dimensions (updated to match table)
        params.image_height = 224
        params.image_width = 224
        params.dynamics_dim = 7
        params.action_dim = 2
        params.continuous_actions = True

    else:
        # Discrete environments (CartPole, SUMO, etc.)
        params.state_dim = env.observation_space.shape[0]
        params.action_dim = env.action_space.n

    # --- Run ---
    runner = ALGOrunner(env, trainer)
    runner.run_experiment(params)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")