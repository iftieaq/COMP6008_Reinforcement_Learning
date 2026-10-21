"""
Q-learning agent.

Off-policy TD control (Watkins and Dayan, 1992; Sutton and Barto, 2018,
section 6.5). The agent behaves epsilon-greedily but learns the value of the
greedy policy: its TD target uses the best Q-value in the next state,
whatever action it will actually take there.

Pseudocode (Sutton and Barto, 2018, section 6.5), with the matching code:

    Initialise Q(s, a) = 0 for all s, a          -> TabularAgent.q_table
    Loop for each episode:                        -> training.train_agent
        Initialise S                              -> env.reset()
        Loop for each step of episode:            -> training.run_training_episode
            Choose A from S using epsilon-greedy  -> select_action()
            Take action A, observe R, S'          -> env.step()
            Q(S, A) <- Q(S, A) + alpha [R + gamma max_a Q(S', a) - Q(S, A)]
                                                  -> learn()
            S <- S'
        until S is terminal
"""

import numpy as np

from agent_common import TabularAgent


class QLearningAgent(TabularAgent):
    """
    Tabular Q-learning agent. See TabularAgent for the constructor arguments.
    """

    def learn(self, state, action, reward, next_state, done):
        """
        Apply the Q-learning update for one transition.

        TD target = reward + gamma * max over a of Q(next_state, a),
        or just the reward if next_state is terminal.

        Args:
            state: state key before the action (S).
            action: action ID taken (A).
            reward: reward received (R).
            next_state: state key after the action (S').
            done: True if S' is terminal.

        Returns:
            Action ID to take in next_state, chosen epsilon-greedily after
            the update (ignored when done is True).
        """
        if done:
            td_target = reward
        else:
            best_next_q_value = np.max(self.get_q_values(next_state))
            td_target = reward + self.discount_factor * best_next_q_value

        self.update_q_value(state, action, td_target)

        return self.select_action(next_state)
