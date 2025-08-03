import json
import random
from typing import Tuple, List, Dict, Optional
import numpy as np
import pygame
import os
import sys

class CellType:
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    AGENT = 4

class GridWorld:
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1)    # RIGHT
    }

    def __init__(self, grid_size: Tuple[int, int], start_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                 obstacles: List[Tuple[int, int]] = None, max_steps_per_episode: int = 100,
                 rewards: Dict[str, float] = None, slip_prob: float = 0.0, 
                 car_sprite: str = "car.png", tile_size: int = 64):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles or []
        self.max_steps_per_episode = max_steps_per_episode
        self.slip_prob = slip_prob
        self.car_sprite_path = car_sprite
        self.tile_size = tile_size
        self.rewards = rewards or {
            'goal': 100.0,
            'obstacle': -100.0,
            'step': -1.0,
            'out_of_bounds': -50.0
        }
        self.n_rows, self.n_cols = grid_size
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.agent_pos = None
        self.episode_step_count = 0
        self._init_grid()
        self._init_pygame()

    def _init_grid(self):
        self.grid.fill(CellType.EMPTY)
        for (r, c) in self.obstacles:
            self.grid[r, c] = CellType.OBSTACLE
        self.grid[self.start_pos] = CellType.START
        self.grid[self.goal_pos] = CellType.GOAL

    def _init_pygame(self):
        pygame.init()
        self.car_img = pygame.image.load(self.car_sprite_path)
        self.car_img = pygame.transform.scale(self.car_img, (self.tile_size, self.tile_size))
        self.colors = {
            CellType.EMPTY: (240, 240, 240),
            CellType.OBSTACLE: (50, 50, 50),
            CellType.START: (100, 230, 100),
            CellType.GOAL: (230, 100, 100),
            CellType.AGENT: (100, 100, 230)
        }
        self.window_size = (self.n_cols * self.tile_size, self.n_rows * self.tile_size)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Autonomous Car Grid World")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_pos
        self.episode_step_count = 0
        return self.agent_pos

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        valid = (0 <= r < self.n_rows) and (0 <= c < self.n_cols)
        if not valid:
            return False
        if self.grid[r, c] == CellType.OBSTACLE:
            return False
        return True

    def get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        valid_actions = []
        for action, (dr, dc) in self.ACTIONS.items():
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(new_pos):
                valid_actions.append(action)
        return valid_actions or list(self.ACTIONS.keys())

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        self.episode_step_count += 1
        if random.random() < self.slip_prob:
            valid_actions = self.get_valid_actions(self.agent_pos)
            if valid_actions:
                action = random.choice(valid_actions)
        dr, dc = self.ACTIONS[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        reward = self.rewards['step']
        done = False
        info = {}
        if new_pos == self.goal_pos:
            reward = self.rewards['goal']
            self.agent_pos = new_pos
            done = True
        elif not self._is_valid_position(new_pos):
            reward = self.rewards['out_of_bounds'] if not (0 <= new_pos[0] < self.n_rows and 0 <= new_pos[1] < self.n_cols) else self.rewards['obstacle']
        else:
            self.agent_pos = new_pos
        if self.episode_step_count >= self.max_steps_per_episode:
            done = True
        return self.agent_pos, reward, done, info

    def render(self, mode: str = "pygame", fps: int = 60, delay: float = 0.2) -> Optional[int]:
        if mode == "pygame":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.window.fill((255, 255, 255))
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                    if (r, c) == self.agent_pos:
                        # Draw agent (car sprite)
                        self.window.blit(self.car_img, (c * self.tile_size, r * self.tile_size))
                    elif (r, c) == self.start_pos:
                        pygame.draw.rect(self.window, self.colors[CellType.START], rect)
                    elif (r, c) == self.goal_pos:
                        pygame.draw.rect(self.window, self.colors[CellType.GOAL], rect)
                    elif (r, c) in self.obstacles:
                        pygame.draw.rect(self.window, self.colors[CellType.OBSTACLE], rect)
                    else:
                        pygame.draw.rect(self.window, self.colors[CellType.EMPTY], rect)
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

            # Draw grid info
            info = f"Steps: {self.episode_step_count}"
            text_surface = self.font.render(info, True, (0, 0, 0))
            self.window.blit(text_surface, (10, 10))
            pygame.display.flip()
            self.clock.tick(fps)
            pygame.time.delay(int(delay * 1000))
        else:
            print("Unsupported render mode.")

    def close(self):
        pygame.quit()

    def save_config(self, filepath: str) -> None:
        config = {
            "grid_size": list(self.grid_size),
            "start": list(self.start_pos),
            "goal": list(self.goal_pos),
            "obstacles": [list(o) for o in self.obstacles]
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_config(cls, filepath: str, **kwargs) -> 'GridWorld':
        with open(filepath) as f:
            config = json.load(f)
        return cls(
            grid_size=tuple(config["grid_size"]),
            start_pos=tuple(config["start"]),
            goal_pos=tuple(config["goal"]),
            obstacles=[tuple(o) for o in config["obstacles"]],
            **kwargs
        )

    def generate_random_obstacles(self, count: int, seed: int = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.obstacles = []
        valid_positions = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)
                           if (r, c) != self.start_pos and (r, c) != self.goal_pos]
        self.obstacles = random.sample(valid_positions, min(count, len(valid_positions)))
        self._init_grid()

    def get_agent_state(self) -> Dict:
        return {
            "position": self.agent_pos,
            "steps": self.episode_step_count
        }
