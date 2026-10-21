"""
Training and evaluation pipeline.

One training loop and one evaluation loop shared by all three agents and the
baseline, so every controller is trained and tested in exactly the same way.

- train_agent() runs many training episodes and records each episode's total
  reward (for learning curves).
- evaluate_controller() runs a controller on a fixed list of test days without
  learning and records per-day results (for comparison tables).
"""

import numpy as np

from environment import discretise_observation

# info keys that are added up over a day to give the daily totals
SUMMED_INFO_KEYS = [
    "grid_import_kwh",
    "grid_export_kwh",
    "grid_cost",
    "battery_charge_kwh",
    "battery_discharge_kwh",
    "ev_charge_kwh",
    "ev_shortfall_kwh",
    "peak_excess_kwh",
    "reward_grid",
    "reward_wear",
    "reward_ev",
    "reward_peak",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training_episode(env, agent):
    """
    Run one training episode (one day), letting the agent learn every step.

    Args:
        env: a HouseholdEnergyEnv (or any Gymnasium environment).
        agent: an agent with select_action, learn and end_episode.

    Returns:
        Total reward collected during the episode.
    """
    observation, info = env.reset()
    state = discretise_observation(observation)
    action = agent.select_action(state)

    total_reward = 0.0
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = discretise_observation(observation)

        action = agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    agent.end_episode()
    return total_reward


def train_agent(env, agent, number_of_episodes, seed):
    """
    Train an agent for a number of episodes.

    The environment is seeded once at the start, so a run with the same seed
    sees the same sequence of training days.

    Args:
        env: a HouseholdEnergyEnv.
        agent: an agent with select_action, learn and end_episode.
        number_of_episodes: how many days to train on.
        seed: seed for the sequence of training days.

    Returns:
        List of total rewards, one per episode (the learning curve).
    """
    env.reset(seed=seed)

    episode_rewards = []
    for episode in range(number_of_episodes):
        episode_rewards.append(run_training_episode(env, agent))
    return episode_rewards


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class GreedyAgentController:
    """
    Wraps a trained agent so it can be evaluated like any other controller.

    It discretises the observation and always picks the greedy action, with no
    exploration and no learning.

    Args:
        agent: a trained agent.
    """

    def __init__(self, agent):
        self.agent = agent

    def act(self, observation):
        """
        Choose the greedy action for an observation.

        Args:
            observation: observation dict from the environment.

        Returns:
            Action ID.
        """
        state = discretise_observation(observation)
        return self.agent.select_action(state, explore=False)


def run_evaluation_episode(env, controller, profile):
    """
    Run one test day with a controller, without any learning.

    Args:
        env: a HouseholdEnergyEnv.
        controller: any object with act(observation) -> action.
        profile: the DayProfile to replay.

    Returns:
        Dict of daily results: total_reward, the daily sum of each key in
        SUMMED_INFO_KEYS, max_grid_import_kwh and ev_target_met.
    """
    observation, info = env.reset(options={"profile": profile})

    day_result = {key: 0.0 for key in SUMMED_INFO_KEYS}
    day_result["total_reward"] = 0.0
    day_result["max_grid_import_kwh"] = 0.0

    done = False
    while not done:
        action = controller.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        day_result["total_reward"] += reward
        for key in SUMMED_INFO_KEYS:
            day_result[key] += info[key]
        day_result["max_grid_import_kwh"] = max(day_result["max_grid_import_kwh"],
                                                info["grid_import_kwh"])

    day_result["ev_target_met"] = day_result["ev_shortfall_kwh"] < 1e-9
    return day_result


def evaluate_controller(env, controller, profiles):
    """
    Evaluate a controller on a list of test days.

    Args:
        env: a HouseholdEnergyEnv.
        controller: any object with act(observation) -> action.
        profiles: list of DayProfile objects (the same list for every controller).

    Returns:
        List of daily result dicts (see run_evaluation_episode).
    """
    return [run_evaluation_episode(env, controller, profile) for profile in profiles]


def average_results(day_results):
    """
    Average each result over all test days.

    Args:
        day_results: list of daily result dicts from evaluate_controller.

    Returns:
        Dict with the mean of every key (ev_target_met becomes a fraction).
    """
    return {key: float(np.mean([day[key] for day in day_results]))
            for key in day_results[0]}


# ---------------------------------------------------------------------------
# Quick check: train Q-learning once and compare it with the baseline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from baseline import RuleBasedController
    from config import DEFAULT_AGENT_SETTINGS, DEFAULT_TRAINING_EPISODES, NUMBER_OF_ACTIONS
    from environment import HouseholdEnergyEnv
    from q_learning import QLearningAgent
    from scenarios import generate_test_profiles

    uncertainty_level = "medium"
    env = HouseholdEnergyEnv(uncertainty_level)
    agent = QLearningAgent(NUMBER_OF_ACTIONS, seed=0, **DEFAULT_AGENT_SETTINGS)

    start_time = time.time()
    rewards = train_agent(env, agent, DEFAULT_TRAINING_EPISODES, seed=0)
    print(f"Training took {time.time() - start_time:.1f} s")
    print(f"States visited: {len(agent.q_table)}")

    block_size = len(rewards) // 10
    for block_start in range(0, len(rewards), block_size):
        block_mean = np.mean(rewards[block_start:block_start + block_size])
        print(f"Episodes {block_start:>6} to {block_start + block_size:>6}: mean reward {block_mean:8.3f}")

    test_profiles = generate_test_profiles(uncertainty_level)
    controllers = {
        "Rule-based baseline": RuleBasedController(),
        "Q-learning": GreedyAgentController(agent),
    }
    for name, controller in controllers.items():
        summary = average_results(evaluate_controller(env, controller, test_profiles))
        print(f"\n{name}")
        for key, value in summary.items():
            print(f"  {key:>22}: {value:8.3f}")
