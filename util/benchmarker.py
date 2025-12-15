import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
import sys


class Utils:
    def __init__(self):
        pass

    def benchmark_plot(self, all_train_returns, all_test_returns, test_interval, all_tests_info, test_episodes,
                       moving_avg_window=100, down_sample_factor=100):
        """Data processing and calculations with comprehensive metrics"""
        num_trials = len(all_train_returns)
        num_points = len(all_test_returns[0])

        # Convert lists to numpy arrays for easier calculations
        all_train_returns = np.array(all_train_returns)
        all_test_returns = np.array(all_test_returns)

        # Calculate the mean and 95% confidence intervals
        mean_train_returns = all_train_returns.mean(axis=0)
        mean_test_returns = all_test_returns.mean(axis=0)

        train_ci = 1.96 * all_train_returns.std(axis=0) / np.sqrt(num_trials)
        test_ci = 1.96 * all_test_returns.std(axis=0) / np.sqrt(num_trials)

        # Calculate individual maximum returns from each trial
        individual_max_returns = [np.max(trial_returns) for trial_returns in all_test_returns]

        # Calculate the average maximum return
        avg_max_return = np.mean(individual_max_returns)

        # Calculate the 95% confidence interval for the average maximum return
        n = len(individual_max_returns)
        sample_std = np.std(individual_max_returns, ddof=1)
        t_value = st.t.ppf(1 - 0.025, df=n - 1)
        margin_of_error = t_value * sample_std / np.sqrt(n)
        avg_max_return_ci = margin_of_error

        """Plot test rewards"""
        plt.figure(figsize=(12, 6))
        episodes = np.arange(0, num_points * test_interval, test_interval)
        for i in range(num_trials):
            plt.plot(episodes, all_test_returns[i], linestyle='dotted', alpha=0.5, label=f'Trial {i+1}')
        plt.plot(episodes, mean_test_returns, '-o', label='Mean Test Returns', color='black')
        plt.fill_between(episodes, mean_test_returns - test_ci, mean_test_returns + test_ci, 
                         color='lightblue', alpha=0.3, label='95% CI')
        plt.xlabel('Episodes')
        plt.ylabel('Test Return')
        plt.title('Test Returns with 95% Confidence Interval')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('test_returns.png', dpi=300)
        plt.show()

        """ Extract and plot comprehensive metrics """
        if all_tests_info is not None and len(all_tests_info) > 0:
            # Extract metrics from all trials
            # Structure: all_tests_info[trial][test_point] = aggregated_info dict
            
            all_distances = []
            all_speeds = []
            all_deviations = []
            all_angles = []
            all_collisions = []
            all_success_rates = []
            all_steering_changes = []
            all_lateral_accels = []

            for trial in all_tests_info:
                trial_distances = [info['avg_distance'] for info in trial]
                trial_speeds = [info['avg_speed'] for info in trial]
                trial_deviations = [info['avg_deviation'] for info in trial]
                trial_angles = [info['avg_angle'] for info in trial]
                trial_collisions = [info['collision_rate'] for info in trial]
                trial_success = [info['success_rate'] for info in trial]
                trial_steering = [info['avg_steering_change'] for info in trial]
                trial_lateral = [info['avg_lateral_accel'] for info in trial]

                all_distances.append(trial_distances)
                all_speeds.append(trial_speeds)
                all_deviations.append(trial_deviations)
                all_angles.append(trial_angles)
                all_collisions.append(trial_collisions)
                all_success_rates.append(trial_success)
                all_steering_changes.append(trial_steering)
                all_lateral_accels.append(trial_lateral)

            # Convert to numpy arrays
            all_distances = np.array(all_distances)
            all_speeds = np.array(all_speeds)
            all_deviations = np.array(all_deviations)
            all_angles = np.array(all_angles)
            all_collisions = np.array(all_collisions)
            all_success_rates = np.array(all_success_rates)
            all_steering_changes = np.array(all_steering_changes)
            all_lateral_accels = np.array(all_lateral_accels)

            # Calculate means and stds
            mean_distances = all_distances.mean(axis=0)
            mean_speeds = all_speeds.mean(axis=0)
            mean_deviations = all_deviations.mean(axis=0)
            mean_angles = all_angles.mean(axis=0)
            mean_collisions = all_collisions.mean(axis=0)
            mean_success = all_success_rates.mean(axis=0)
            mean_steering = all_steering_changes.mean(axis=0)
            mean_lateral = all_lateral_accels.mean(axis=0)

            std_distances = all_distances.std(axis=0)
            std_speeds = all_speeds.std(axis=0)
            std_deviations = all_deviations.std(axis=0)
            std_angles = all_angles.std(axis=0)
            std_collisions = all_collisions.std(axis=0)
            std_success = all_success_rates.std(axis=0)
            std_steering = all_steering_changes.std(axis=0)
            std_lateral = all_lateral_accels.std(axis=0)

            # Create comprehensive summary table
            table_episodes = episodes[::10]  # Every 100 episodes
            
            summary_df = pd.DataFrame({
                'Episode': table_episodes,
                'Distance (m)': [f"{d:.2f}±{s:.2f}" for d, s in zip(mean_distances[::10], std_distances[::10])],
                'Speed (m/s)': [f"{sp:.2f}±{st:.2f}" for sp, st in zip(mean_speeds[::10], std_speeds[::10])],
                'Speed (km/h)': [f"{sp*3.6:.2f}±{st*3.6:.2f}" for sp, st in zip(mean_speeds[::10], std_speeds[::10])],
                'Deviation (m)': [f"{d:.2f}±{s:.2f}" for d, s in zip(mean_deviations[::10], std_deviations[::10])],
                'Angle (deg)': [f"{a:.2f}±{s:.2f}" for a, s in zip(mean_angles[::10], std_angles[::10])],
                'Collision (%)': [f"{c:.2f}±{s:.2f}" for c, s in zip(mean_collisions[::10], std_collisions[::10])],
                'Success (%)': [f"{sr:.2f}±{s:.2f}" for sr, s in zip(mean_success[::10], std_success[::10])],
            })

            # Print comprehensive table
            print("\n" + "=" * 150)
            print("COMPREHENSIVE METRICS ACROSS ALL TRIALS (Every 100 Episodes)")
            print("=" * 150)
            print(summary_df.to_string(index=False))
            print("=" * 150)

            # Print final metrics summary (Paper Table VI style)
            print("\n" + "=" * 100)
            print(f"FINAL METRICS SUMMARY (Episode {int(episodes[-1])})")
            print("=" * 100)
            print(f"Distance Travelled:    {mean_distances[-1]:.2f} ± {std_distances[-1]:.2f} m")
            print(f"Average Speed:         {mean_speeds[-1]:.2f} ± {std_speeds[-1]:.2f} m/s "
                  f"({mean_speeds[-1]*3.6:.2f} ± {std_speeds[-1]*3.6:.2f} km/h)")
            print(f"Average Deviation:     {mean_deviations[-1]:.2f} ± {std_deviations[-1]:.2f} m")
            print(f"Average Angle Error:   {mean_angles[-1]:.2f} ± {std_angles[-1]:.2f} degrees")
            print(f"Collision Rate:        {mean_collisions[-1]:.2f} ± {std_collisions[-1]:.2f} %")
            print(f"Success Rate:          {mean_success[-1]:.2f} ± {std_success[-1]:.2f} %")
            print(f"Steering Smoothness:   {mean_steering[-1]:.4f} ± {std_steering[-1]:.4f}")
            print(f"Lateral Acceleration:  {mean_lateral[-1]:.2f} ± {std_lateral[-1]:.2f} m/s²")
            print("=" * 100)

            # Print initial metrics
            print("\n" + "=" * 100)
            print(f"INITIAL METRICS SUMMARY (Episode {int(episodes[0])})")
            print("=" * 100)
            print(f"Distance Travelled:    {mean_distances[0]:.2f} ± {std_distances[0]:.2f} m")
            print(f"Average Speed:         {mean_speeds[0]:.2f} ± {std_speeds[0]:.2f} m/s "
                  f"({mean_speeds[0]*3.6:.2f} ± {std_speeds[0]*3.6:.2f} km/h)")
            print(f"Average Deviation:     {mean_deviations[0]:.2f} ± {std_deviations[0]:.2f} m")
            print(f"Average Angle Error:   {mean_angles[0]:.2f} ± {std_angles[0]:.2f} degrees")
            print(f"Collision Rate:        {mean_collisions[0]:.2f} ± {std_collisions[0]:.2f} %")
            print(f"Success Rate:          {mean_success[0]:.2f} ± {std_success[0]:.2f} %")
            print("=" * 100)

            # Print improvement metrics
            print("\n" + "=" * 100)
            print("IMPROVEMENT ANALYSIS")
            print("=" * 100)
            dist_change = mean_distances[-1] - mean_distances[0]
            speed_change = mean_speeds[-1] - mean_speeds[0]
            dev_change = mean_deviations[-1] - mean_deviations[0]
            angle_change = mean_angles[-1] - mean_angles[0]
            coll_change = mean_collisions[-1] - mean_collisions[0]
            success_change = mean_success[-1] - mean_success[0]
            
            print(f"Distance:     {dist_change:+.2f} m ({dist_change/mean_distances[0]*100:+.1f}%)")
            print(f"Speed:        {speed_change:+.2f} m/s ({speed_change/mean_speeds[0]*100:+.1f}%)")
            print(f"Deviation:    {dev_change:+.2f} m ({dev_change/mean_deviations[0]*100:+.1f}%)")
            print(f"Angle Error:  {angle_change:+.2f} deg ({angle_change/mean_angles[0]*100:+.1f}%)")
            print(f"Collision:    {coll_change:+.2f}% ({coll_change/mean_collisions[0]*100:+.1f}%)")
            print(f"Success:      {success_change:+.2f}% ({success_change/mean_success[0]*100:+.1f}%)")
            print("=" * 100)

            # Plot Paper Figure 8 style metrics
            self._plot_training_progression(episodes, mean_speeds, std_speeds, 
                                           mean_deviations, std_deviations,
                                           mean_angles, std_angles,
                                           mean_test_returns, test_ci)

        # Plot density plot of test returns
        plt.figure(figsize=(12, 6))
        sns.kdeplot(mean_test_returns, label='Density Plot', fill=True)
        plt.xlabel('Test Return')
        plt.ylabel('Density')
        plt.title('Density Plot of Test Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('density_plot.png', dpi=300)
        plt.show()

        return mean_test_returns, avg_max_return, avg_max_return_ci, individual_max_returns

    def _plot_training_progression(self, episodes, mean_speeds, std_speeds,
                                   mean_deviations, std_deviations,
                                   mean_angles, std_angles,
                                   mean_rewards, reward_ci):
        """Plot training progression similar to Paper Figure 8"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average Reward
        ax = axes[0, 0]
        ax.plot(episodes, mean_rewards, '-o', color='blue', label='Mean')
        ax.fill_between(episodes, mean_rewards - reward_ci, mean_rewards + reward_ci,
                       alpha=0.3, color='blue', label='95% CI')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Reward per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Average Speed
        ax = axes[0, 1]
        ax.plot(episodes, mean_speeds, '-o', color='green', label='Mean')
        ax.fill_between(episodes, mean_speeds - std_speeds, mean_speeds + std_speeds,
                       alpha=0.3, color='green', label='Std Dev')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Speed (m/s)')
        ax.set_title('Average Speed per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Average Deviation
        ax = axes[1, 0]
        ax.plot(episodes, mean_deviations, '-o', color='orange', label='Mean')
        ax.fill_between(episodes, mean_deviations - std_deviations, mean_deviations + std_deviations,
                       alpha=0.3, color='orange', label='Std Dev')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Deviation (m)')
        ax.set_title('Average Deviation per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Average Angle
        ax = axes[1, 1]
        ax.plot(episodes, mean_angles, '-o', color='red', label='Mean')
        ax.fill_between(episodes, mean_angles - std_angles, mean_angles + std_angles,
                       alpha=0.3, color='red', label='Std Dev')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Angle Error (deg)')
        ax.set_title('Average Angle Error per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progression.png', dpi=300)
        plt.show()