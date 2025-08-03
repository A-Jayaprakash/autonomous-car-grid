import numpy as np
import random
from typing import Tuple, List, Dict, Optional
import pickle
import os

class QLearningAgent:
    def __init__(self, state_size: Tuple[int, int], n_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 initial_epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01, seed: int = None):
        self.n_rows, self.n_cols = state_size
        self.n_actions = n_actions
        self.Q = np.zeros((self.n_rows, self.n_cols, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def policy(self, state: Tuple[int, int], valid_actions: Optional[List[int]] = None, goal: Tuple[int, int] = None) -> int:
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.Q[state[0], state[1], a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, v in zip(valid_actions, q_values) if v == max_q]
            if len(best_actions) > 1 and goal is not None:
                # Tiebreak using Manhattan distance to the goal
                min_dist = float('inf')
                choice = None
                for a in best_actions:
                    dr, dc = GridWorld.ACTIONS[a]
                    new_pos = (state[0] + dr, state[1] + dc)
                    dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
                    if dist < min_dist:
                        min_dist = dist
                        choice = a
                return choice
            return best_actions[0]

    def learn(self, state: Tuple[int, int], action: int, reward: float,
              next_state: Optional[Tuple[int, int]], next_valid_actions: List[int]) -> None:
        current_q = self.Q[state[0], state[1], action]
        if next_state is None:  # terminal state
            target = reward
        else:
            max_next_q = max([self.Q[next_state[0], next_state[1], a] for a in next_valid_actions], default=0)
            target = reward + self.discount_factor * max_next_q
        self.Q[state[0], state[1], action] += self.learning_rate * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def save_model(self, filename: str = "q_table.npy") -> None:
        np.save(filename, self.Q)

    def load_model(self, filename: str = "q_table.npy") -> bool:
        if os.path.exists(filename):
            self.Q = np.load(filename)
            return True
        return False


# Since environment.py refers to GridWorld.ACTIONS in the agent,
# if importing policy method for tie-breaker, define a stub for illustration:
try:
    from environment import GridWorld
except ImportError:
    class GridWorld:
        ACTIONS = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
        }

