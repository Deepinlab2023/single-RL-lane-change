import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
import sys
#TODO: Fix confidence interval / benchmarking calculation

class Utils:
    def __init__(self):
        pass

    def benchmark_plot(self, all_train_returns, all_test_returns, test_interval, all_tests_info, test_episodes,
                       moving_avg_window=100, down_sample_factor=100):
        """Data processing and calculations"""
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

        # # Apply moving average to smooth the training returns
        # smoothed_mean_train_returns = np.convolve(mean_train_returns, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        # smoothed_train_ci = np.convolve(train_ci, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        #
        # # Down-sample the training returns for plotting
        # down_sampled_indices = np.arange(0, len(smoothed_mean_train_returns), down_sample_factor)
        # down_sampled_mean_train_returns = smoothed_mean_train_returns[down_sampled_indices]
        # down_sampled_train_ci = smoothed_train_ci[down_sampled_indices]


        """Plot test rewards"""
        plt.figure(figsize=(12, 6))
        episodes = np.arange(0, num_points * test_interval, test_interval)
        for i in range(num_trials):
            plt.plot(episodes, all_test_returns[i], linestyle='dotted', alpha=0.5, label=f'Trial {i+1}')  # Individual test trials
        plt.plot(episodes, mean_test_returns, '-o', label='Mean Test Returns', color='black')  # Mean test returns without error bars
        plt.fill_between(episodes, mean_test_returns - test_ci, mean_test_returns + test_ci, color='lightblue', alpha=0.3, label='CI')  # Fill between upper and lower bounds
        plt.xlabel('Episodes')
        plt.ylabel('Test Return')
        plt.title('Test Returns with 95% Confidence Interval')
        plt.legend()
        plt.show()
        
        """ Plot Information Table """
        # Extract and create summary table from all_tests_info
        if all_tests_info is not None and len(all_tests_info) > 0:
            # Extract metrics: all_tests_info[trial][test_idx] = (distance, speed, collision_pct)
            all_distances = []
            all_speeds = []
            all_collisions = []

            for trial in all_tests_info:
                trial_distances = [info[0] for info in trial]
                trial_speeds = [info[1] for info in trial]
                trial_collisions = [info[2] for info in trial]

                all_distances.append(trial_distances)
                all_speeds.append(trial_speeds)
                all_collisions.append(trial_collisions)

            all_distances = np.array(all_distances)
            all_speeds = np.array(all_speeds)
            all_collisions = np.array(all_collisions)

            mean_distances = all_distances.mean(axis=0)
            mean_speeds = all_speeds.mean(axis=0)
            mean_collisions = all_collisions.mean(axis=0)

            std_distances = all_distances.std(axis=0)
            std_speeds = all_speeds.std(axis=0)
            std_collisions = all_collisions.std(axis=0)

            # Create summary table (show every 100 episodes)
            table_episodes = episodes[::10]  # Every 100 episodes (since test_interval=10, ::10 gives every 100)
            table_distances = mean_distances[::10]
            table_speeds = mean_speeds[::10]
            table_collisions = mean_collisions[::10]
            table_dist_std = std_distances[::10]
            table_speed_std = std_speeds[::10]
            table_coll_std = std_collisions[::10]

            # Create DataFrame
            summary_df = pd.DataFrame({
                'Episode': table_episodes,
                'Avg Distance (m)': [f"{d:.2f} ± {s:.2f}" for d, s in zip(table_distances, table_dist_std)],
                'Avg Speed (m/s)': [f"{sp:.2f} ± {st:.2f}" for sp, st in zip(table_speeds, table_speed_std)],
                'Avg Speed (km/h)': [f"{sp * 3.6:.2f} ± {st * 3.6:.2f}" for sp, st in
                                     zip(table_speeds, table_speed_std)],
                'Avg Collision (%)': [f"{c:.2f} ± {s:.2f}" for c, s in zip(table_collisions, table_coll_std)]
            })

            # Print table
            print("\n" + "=" * 100)
            print("AVERAGED METRICS ACROSS ALL TRIALS (Every 100 Episodes)")
            print("=" * 100)
            print(summary_df.to_string(index=False))
            print("=" * 100)

            # Print summary statistics
            print("\n=== Final Metrics Summary (Episode {}) ===".format(int(episodes[-1])))
            print(f"Average Distance: {mean_distances[-1]:.2f} ± {std_distances[-1]:.2f} m")
            print(
                f"Average Speed: {mean_speeds[-1]:.2f} ± {std_speeds[-1]:.2f} m/s ({mean_speeds[-1] * 3.6:.2f} ± {std_speeds[-1] * 3.6:.2f} km/h)")
            print(f"Average Collision Rate: {mean_collisions[-1]:.2f} ± {std_collisions[-1]:.2f}%")

            print("\n=== Initial Metrics Summary (Episode {}) ===".format(int(episodes[0])))
            print(f"Average Distance: {mean_distances[0]:.2f} ± {std_distances[0]:.2f} m")
            print(
                f"Average Speed: {mean_speeds[0]:.2f} ± {std_speeds[0]:.2f} m/s ({mean_speeds[0] * 3.6:.2f} ± {std_speeds[0] * 3.6:.2f} km/h)")
            print(f"Average Collision Rate: {mean_collisions[0]:.2f} ± {std_collisions[0]:.2f}%")

            print("\n=== Improvement ===")
            print(
                f"Distance: +{mean_distances[-1] - mean_distances[0]:.2f} m ({((mean_distances[-1] - mean_distances[0]) / mean_distances[0] * 100):.1f}%)")
            print(
                f"Speed: +{mean_speeds[-1] - mean_speeds[0]:.2f} m/s ({((mean_speeds[-1] - mean_speeds[0]) / mean_speeds[0] * 100):.1f}%)")
            print(
                f"Collision Rate: {mean_collisions[-1] - mean_collisions[0]:.2f}% ({((mean_collisions[-1] - mean_collisions[0]) / mean_collisions[0] * 100):.1f}%)")



        # Plot density plot of test returns
        plt.figure(figsize=(12, 6))
        #sns.kdeplot(mean_test_returns, fill=True, label='Density Plot')
        sns.kdeplot(mean_test_returns, label='Density Plot')
        plt.xlabel('Test Return')
        plt.ylabel('Density')
        plt.title('Density Plot of Test Returns')
        plt.legend()
        plt.show()

        """Plot test rewards (Not in use)"""
        # # Plot training returns with moving average and confidence interval
        # plt.figure(figsize=(12, 6))
        # plt.plot(down_sampled_indices, down_sampled_mean_train_returns, label='Mean Training Returns (Smoothed)', color='blue')
        # plt.fill_between(down_sampled_indices, down_sampled_mean_train_returns - down_sampled_train_ci, down_sampled_mean_train_returns + down_sampled_train_ci, color='lightblue', alpha=0.3, label='CI')
        # plt.xlabel('Episodes')
        # plt.ylabel('Training Return')
        # plt.title('Training Returns with 95% Confidence Interval (Smoothed)')
        # plt.legend()
        # plt.show()

        # # Plot density plot of training returns
        # plt.figure(figsize=(12, 6))
        # #sns.kdeplot(mean_train_returns, fill=True, label='Density Plot')
        # sns.kdeplot(mean_train_returns, label='Density Plot')
        # plt.xlabel('Training Return')
        # plt.ylabel('Density')
        # plt.title('Density Plot of Training Returns')
        # plt.legend()
        # plt.show()
        
        return mean_test_returns, avg_max_return, avg_max_return_ci, individual_max_returns