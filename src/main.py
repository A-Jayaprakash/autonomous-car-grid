from environment import GridWorld
from agent import QLearningAgent

import pygame
import random

def main():
    env = GridWorld(
        grid_size=(5, 5),
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 2), (3, 3)],
        slip_prob=0.1,
        car_sprite="car.png",
        tile_size=96
    )

    agent = QLearningAgent(
        state_size=env.grid_size,
        n_actions=len(env.ACTIONS),
        learning_rate=0.1,
        discount_factor=0.9,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    n_episodes = 1000
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            valid_actions = env.get_valid_actions(state)
            action = agent.policy(state, valid_actions, goal=env.goal_pos)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions(next_state) if not done else []
            agent.learn(state, action, reward, next_state, next_valid_actions)
            state = next_state
            episode_reward += reward
            env.render(mode="pygame", fps=10, delay=0.1)  # Optional: Remove for faster training
            if done:
                break
        agent.decay_epsilon()
        print(f"Episode {episode}: reward={episode_reward}, epsilon={agent.epsilon}")

    agent.save_model("q_table.npy")
    env.close()

if __name__ == "__main__":
    main()
