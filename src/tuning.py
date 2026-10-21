"""
Hyperparameter tuning.

A one-at-a-time sweep that works for any agent class with the shared agent
interface, so Q-learning, SARSA and Expected SARSA are all tuned by exactly
the same procedure.

Procedure:
1. Start from DEFAULT_AGENT_SETTINGS.
2. Change one hyperparameter at a time to each of its candidate values,
   keeping the others at their defaults.
3. Train every configuration with several seeds.
4. Score each trained agent on validation days. These come from their own
   seed range, separate from training and from the test days, so the test
   days are never used for tuning.
5. Combine the best value of each hyperparameter into one configuration.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from config import DEFAULT_AGENT_SETTINGS, DEFAULT_TRAINING_EPISODES, NUMBER_OF_ACTIONS
from environment import HouseholdEnergyEnv
from scenarios import generate_fixed_profiles
from training import GreedyAgentController, average_results, evaluate_controller, train_agent

VALIDATION_SEED_START = 2_000_000
NUMBER_OF_VALIDATION_DAYS = 100

# epsilon_decay values make epsilon reach 0.05 after 25%, 50% or 75% of 400,000 episodes
TUNING_CANDIDATES = {
    "learning_rate": [0.05, 0.1, 0.2],
    "discount_factor": [0.90, 0.95, 0.99],
    "epsilon_decay": [0.99997, 0.999985, 0.99999],
}

TUNING_SEEDS = [0, 1, 2]
LEARNING_CURVE_POINTS = 100


# ---------------------------------------------------------------------------
# Building the sweep
# ---------------------------------------------------------------------------

def generate_validation_profiles(uncertainty_level, number_of_days=NUMBER_OF_VALIDATION_DAYS):
    """
    Generate the fixed validation days used to score tuning runs.

    Args:
        uncertainty_level: "low", "medium" or "high".
        number_of_days: how many validation days to generate.

    Returns:
        List of DayProfile objects, seeded from VALIDATION_SEED_START.
    """
    return generate_fixed_profiles(uncertainty_level, number_of_days, VALIDATION_SEED_START)


def one_at_a_time_configurations(default_settings, candidates):
    """
    List the configurations of a one-at-a-time sweep.

    Args:
        default_settings: dict of default hyperparameters.
        candidates: dict mapping a hyperparameter name to the values to try.

    Returns:
        List of (configuration name, settings dict). The defaults appear once,
        named "default"; every other configuration differs from the defaults
        in exactly one hyperparameter, e.g. "learning_rate=0.05".
    """
    configurations = [("default", dict(default_settings))]

    for parameter_name, values in candidates.items():
        for value in values:
            if value == default_settings[parameter_name]:
                continue
            settings = dict(default_settings)
            settings[parameter_name] = value
            configurations.append((f"{parameter_name}={value}", settings))

    return configurations


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def summarise_learning_curve(episode_rewards, number_of_points=LEARNING_CURVE_POINTS):
    """
    Shrink a learning curve to a fixed number of points by averaging blocks.

    Args:
        episode_rewards: list of total rewards, one per training episode.
        number_of_points: how many averaged points to keep.

    Returns:
        List of number_of_points block averages.
    """
    blocks = np.array_split(np.array(episode_rewards), number_of_points)
    return [float(block.mean()) for block in blocks]


def run_configuration(agent_class, settings, seed, uncertainty_level,
                      number_of_episodes, validation_profiles):
    """
    Train one agent with one configuration and seed, then score it.

    Args:
        agent_class: agent class to build, e.g. QLearningAgent.
        settings: dict of hyperparameters passed to the agent.
        seed: seed for both the agent and the training days.
        uncertainty_level: "low", "medium" or "high".
        number_of_episodes: number of training episodes.
        validation_profiles: list of DayProfile objects to score on.

    Returns:
        Dict with the seed, the averaged validation results, the number of
        states visited and the summarised learning curve.
    """
    env = HouseholdEnergyEnv(uncertainty_level)
    agent = agent_class(NUMBER_OF_ACTIONS, seed=seed, **settings)

    episode_rewards = train_agent(env, agent, number_of_episodes, seed)
    validation_results = evaluate_controller(env, GreedyAgentController(agent), validation_profiles)

    run_result = average_results(validation_results)
    run_result["seed"] = seed
    run_result["states_visited"] = len(agent.q_table)
    run_result["learning_curve"] = summarise_learning_curve(episode_rewards)
    return run_result


def run_many(jobs, parallel_workers):
    """
    Run a list of run_configuration jobs, in parallel if requested.

    Args:
        jobs: list of argument tuples for run_configuration.
        parallel_workers: number of processes; 1 runs the jobs one by one.

    Returns:
        List of run results, in the same order as jobs.
    """
    if parallel_workers == 1:
        return [run_configuration(*job) for job in jobs]

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(run_configuration, *job) for job in jobs]
        return [future.result() for future in futures]


def run_tuning(agent_class, uncertainty_level="medium", seeds=TUNING_SEEDS,
               number_of_episodes=DEFAULT_TRAINING_EPISODES, parallel_workers=None):
    """
    Run the one-at-a-time sweep for one agent class.

    Args:
        agent_class: agent class to tune.
        uncertainty_level: uncertainty level for training and validation.
        seeds: list of seeds; every configuration is trained once per seed.
        number_of_episodes: training episodes per run.
        parallel_workers: number of processes (None uses all CPU cores, 1
            runs one by one).

    Returns:
        List of dicts, one per configuration, each with "name", "settings"
        and "runs" (the list of run results, one per seed).
    """
    configurations = one_at_a_time_configurations(DEFAULT_AGENT_SETTINGS, TUNING_CANDIDATES)
    validation_profiles = generate_validation_profiles(uncertainty_level)

    jobs = []
    for name, settings in configurations:
        for seed in seeds:
            jobs.append((agent_class, settings, seed, uncertainty_level,
                         number_of_episodes, validation_profiles))

    run_results = run_many(jobs, parallel_workers)

    tuning_results = []
    for index, (name, settings) in enumerate(configurations):
        first_run = index * len(seeds)
        runs = run_results[first_run:first_run + len(seeds)]
        tuning_results.append({"name": name, "settings": settings, "runs": runs})
    return tuning_results


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------

def summarise_configuration(configuration_result):
    """
    Average one configuration's results over its seeds.

    Args:
        configuration_result: one entry from run_tuning.

    Returns:
        Dict with the name, settings, the mean and standard deviation of the
        validation reward, and the means of the main operational results.
    """
    runs = configuration_result["runs"]
    rewards = [run["total_reward"] for run in runs]

    return {
        "name": configuration_result["name"],
        "settings": configuration_result["settings"],
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "grid_cost": float(np.mean([run["grid_cost"] for run in runs])),
        "ev_target_met": float(np.mean([run["ev_target_met"] for run in runs])),
        "battery_charge_kwh": float(np.mean([run["battery_charge_kwh"] for run in runs])),
        "states_visited": float(np.mean([run["states_visited"] for run in runs])),
    }


def choose_best_settings(summaries, default_settings, candidates):
    """
    Combine the best value of each hyperparameter into one configuration.

    For each hyperparameter, compare the default configuration with the
    configurations that changed only that hyperparameter, and keep the value
    with the highest mean validation reward.

    Args:
        summaries: list of dicts from summarise_configuration.
        default_settings: dict of default hyperparameters.
        candidates: dict of the values that were tried.

    Returns:
        Dict of the chosen hyperparameters.
    """
    default_summary = next(summary for summary in summaries if summary["name"] == "default")
    best_settings = dict(default_settings)

    for parameter_name in candidates:
        best_summary = default_summary
        for summary in summaries:
            changed_only_this = summary["name"].startswith(parameter_name + "=")
            if changed_only_this and summary["mean_reward"] > best_summary["mean_reward"]:
                best_summary = summary
        best_settings[parameter_name] = best_summary["settings"][parameter_name]

    return best_settings


def print_tuning_table(summaries):
    """
    Print the tuning summaries as a table.

    Args:
        summaries: list of dicts from summarise_configuration.
    """
    print(f"{'Configuration':<24}{'Reward (mean ± std)':>22}{'Cost $':>9}"
          f"{'EV met':>9}{'Battery kWh':>13}{'States':>9}")
    for summary in summaries:
        reward_text = f"{summary['mean_reward']:.3f} ± {summary['std_reward']:.3f}"
        print(f"{summary['name']:<24}{reward_text:>22}{summary['grid_cost']:>9.3f}"
              f"{summary['ev_target_met']:>9.3f}{summary['battery_charge_kwh']:>13.2f}"
              f"{summary['states_visited']:>9.0f}")


# ---------------------------------------------------------------------------
# Quick check: tune Q-learning and confirm the chosen settings
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from q_learning import QLearningAgent

    from baseline import RuleBasedController

    baseline_results = evaluate_controller(HouseholdEnergyEnv("medium"), RuleBasedController(),
                                           generate_validation_profiles("medium"))
    print(f"Baseline on the same validation days: reward {average_results(baseline_results)['total_reward']:.3f}\n")

    start_time = time.time()
    tuning_results = run_tuning(QLearningAgent)
    summaries = [summarise_configuration(result) for result in tuning_results]
    print_tuning_table(summaries)

    best_settings = choose_best_settings(summaries, DEFAULT_AGENT_SETTINGS, TUNING_CANDIDATES)
    print(f"\nChosen settings: {best_settings}")

    confirmation_seeds = [10, 11, 12, 13, 14]
    validation_profiles = generate_validation_profiles("medium")
    jobs = [(QLearningAgent, best_settings, seed, "medium", DEFAULT_TRAINING_EPISODES, validation_profiles)
            for seed in confirmation_seeds]
    confirmation = summarise_configuration({"name": "chosen",
                                            "settings": best_settings,
                                            "runs": run_many(jobs, parallel_workers=None)})
    print_tuning_table([confirmation])
    print(f"\nTotal time: {time.time() - start_time:.0f} s")
