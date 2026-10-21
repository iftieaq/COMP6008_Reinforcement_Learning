"""
Shared components of the three tabular agents.

TabularAgent holds everything Q-learning, SARSA and Expected SARSA have in
common: the Q-table, epsilon-greedy action selection, epsilon decay and the
basic TD update step. Each agent subclasses it and only adds its own learn()
method, which is where the three algorithms differ (how they build the TD
target). Keeping everything else identical makes the comparison fair.

Initial version by Person 1 so the Q-learning agent can run. Person 2 owns
this file and may extend it without changing the interface.
"""

import numpy as np


class TabularAgent:
    """
    Base class for tabular TD control agents.

    Symbols used in the docstrings:
        alpha (learning_rate): how far each update moves a Q-value towards
            its target, between 0 and 1.
        gamma (discount_factor): how much future rewards count, between 0 and 1.
        epsilon: probability of picking a random action while training.

    Args:
        number_of_actions: size of the action space.
        learning_rate: alpha.
        discount_factor: gamma.
        epsilon_start: epsilon at the start of training.
        epsilon_decay: epsilon is multiplied by this after every episode.
        epsilon_min: epsilon never drops below this.
        seed: seed for the agent's own random number generator.
    """

    def __init__(self, number_of_actions, learning_rate, discount_factor,
                 epsilon_start, epsilon_decay, epsilon_min, seed=None):
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.random_generator = np.random.default_rng(seed)

        # State key -> array of Q-values, one per action. Missing states count as all zeros.
        self.q_table = {}
        # State key -> array of update counts, one per action (for analysing coverage)
        self.visit_counts = {}

    def get_q_values(self, state):
        """
        Return the Q-values for a state without changing the table.

        Args:
            state: hashable state key.

        Returns:
            Array of number_of_actions Q-values (all zeros if never updated).
        """
        if state in self.q_table:
            return self.q_table[state]
        return np.zeros(self.number_of_actions)

    def greedy_action(self, state):
        """
        Pick the action with the highest Q-value, breaking ties at random.

        Random tie-breaking matters early in training, when all Q-values are
        zero and a fixed choice would always pick action 0.

        Args:
            state: hashable state key.

        Returns:
            Action ID.
        """
        q_values = self.get_q_values(state)
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(self.random_generator.choice(best_actions))

    def select_action(self, state, explore=True):
        """
        Choose an action with epsilon-greedy.

        Args:
            state: hashable state key.
            explore: if True, pick a random action with probability epsilon.
                If False (evaluation), always act greedily.

        Returns:
            Action ID.
        """
        if explore and self.random_generator.random() < self.epsilon:
            return int(self.random_generator.integers(self.number_of_actions))
        return self.greedy_action(state)

    def update_q_value(self, state, action, td_target):
        """
        Move Q(state, action) a step towards the TD target.

        Q(s, a) <- Q(s, a) + alpha * (td_target - Q(s, a))

        Args:
            state: hashable state key.
            action: action ID that was taken.
            td_target: the agent's target value for Q(s, a).

        Returns:
            The TD error (td_target - old Q-value), useful for analysis.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.number_of_actions)
        if state not in self.visit_counts:
            self.visit_counts[state] = np.zeros(self.number_of_actions, dtype=int)

        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        self.visit_counts[state][action] += 1
        return td_error

    def learn(self, state, action, reward, next_state, done):
        """
        Learn from one transition and return the next action to take.

        Each agent implements this with its own update rule.

        Args:
            state: state key before the action.
            action: action ID taken.
            reward: reward received.
            next_state: state key after the action.
            done: True if next_state ends the episode.

        Returns:
            Action ID to take in next_state (ignored when done is True).
        """
        raise NotImplementedError("Each agent must implement learn().")

    def end_episode(self):
        """
        Decay epsilon at the end of an episode, never going below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
