import argparse
import time
import gymnasium as gym
from envs.sumo_env import SumoEnv  # ensures SUMO env is imported and registered

from runner.runner import ALGOrunner
from train.ppo_trainer import PPOtrainer
from train.dqn_trainer import DQNtrainer
from train.a2c_trainer import A2Ctrainer
from util.parameters import ParametersPPO, ParametersDQN, ParametersA2C


def main():
    parser = argparse.ArgumentParser(
        description="Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True,
                        help='Choose: "cartpole", "pong", or "sumo".')
    parser.add_argument('--algo', type=str, required=True,
                        help='Choose: "dqn", "ppo", or "a2c".')
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
        env = SumoEnv(max_steps=40,v_max=30,render_mode=None ,step_length=0.2,decision_steps=5, w_collision=0.5,w_efficiency=0.1,w_lane_change=0.1)
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # --- Select algorithm ---
    if args.algo.lower() == 'ppo':
        params = ParametersPPO()    
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