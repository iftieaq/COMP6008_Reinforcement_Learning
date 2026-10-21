"""
Checks for the Q-learning agent: the update rule on a hand-worked example,
and learning the optimal path on Gymnasium's CliffWalking grid world.
"""

import gymnasium as gym
import numpy as np
import pytest

from q_learning import QLearningAgent


def make_agent(number_of_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, seed=0):
    """
    Create a Q-learning agent with a fixed epsilon (no decay).

    Args:
        number_of_actions: size of the action space.
        learning_rate: alpha.
        discount_factor: gamma.
        epsilon: exploration rate.
        seed: random seed.

    Returns:
        QLearningAgent.
    """
    return QLearningAgent(number_of_actions, learning_rate, discount_factor,
                          epsilon_start=epsilon, epsilon_decay=1.0, epsilon_min=epsilon, seed=seed)


def test_update_matches_hand_worked_example():
    # Three actions: 0 idle, 1 charge, 2 discharge
    agent = make_agent(number_of_actions=3)
    agent.q_table["noon"] = np.array([0.0, -2.0, 0.0])
    agent.q_table["half_past_noon"] = np.array([-1.5, -1.8, -0.9])

    agent.learn("noon", action=1, reward=0.01, next_state="half_past_noon", done=False)

    # target = 0.01 + 0.95 * max(-1.5, -1.8, -0.9) = -0.845
    # new Q  = -2.0 + 0.1 * (-0.845 - (-2.0)) = -1.8845
    assert agent.q_table["noon"][1] == pytest.approx(-1.8845)


def test_terminal_update_uses_reward_only():
    agent = make_agent(number_of_actions=3)
    agent.q_table["last_step"] = np.array([0.0, -2.0, 0.0])
    agent.q_table["end_of_day"] = np.array([100.0, 100.0, 100.0])

    agent.learn("last_step", action=1, reward=-1.0, next_state="end_of_day", done=True)

    # target = -1.0, so new Q = -2.0 + 0.1 * (-1.0 - (-2.0)) = -1.9
    assert agent.q_table["last_step"][1] == pytest.approx(-1.9)


def test_learns_shortest_path_on_cliff_walking():
    env = gym.make("CliffWalking-v1")
    agent = make_agent(env.action_space.n, learning_rate=0.5, discount_factor=1.0, epsilon=0.1)

    state, info = env.reset(seed=0)
    for episode in range(500):
        state, info = env.reset()
        action = agent.select_action(state)
        done = False
        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            action = agent.learn(state, action, reward, next_state, done)
            state = next_state

    # Greedy run: the optimal path along the cliff edge takes 13 steps of -1 each
    state, info = env.reset()
    total_reward = 0
    for step in range(100):
        state, reward, terminated, truncated, info = env.step(agent.select_action(state, explore=False))
        total_reward += reward
        if terminated or truncated:
            break
    assert total_reward == -13
