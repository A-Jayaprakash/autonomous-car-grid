import argparse
from environment import GridWorld
from agent import QLearningAgent

def main():
    parser = argparse.ArgumentParser(description="Visualize the optimal path of the trained agent.")
    parser.add_argument(
        "--config",
        default="grid_config.json",
        help="Path to grid configuration JSON file."
    )
    parser.add_argument(
        "--car",
        default="car.png",
        help="Path to the car sprite image file (top-down, transparent background)."
    )
    parser.add_argument(
        "--tile",
        default=96,
        type=int,
        help="Tile size for Pygame renderer."
    )
    parser.add_argument(
        "--max_steps",
        default=50,
        type=int,
        help="Max steps to visualize before giving up."
    )
    args = parser.parse_args()

    # Load the environment
    env = GridWorld.load_config(
        args.config,
        car_sprite=args.car,
        tile_size=args.tile
    )

    # Initialize and load the agent
    agent = QLearningAgent(
        state_size=env.grid_size,
        n_actions=len(env.ACTIONS)
    )
    agent.load_model("q_table.npy")
    agent.epsilon = 0.0  # Force greedy (optimal) behavior

    # Run one episode and visualize
    state = env.reset()
    done = False
    step_count = 0
    path = [state]
    goal = env.goal_pos

    while not done and step_count < args.max_steps:
        valid_actions = env.get_valid_actions(state)
        action = agent.policy(state, valid_actions, goal=goal)
        state, reward, done, info = env.step(action)
        path.append(state)
        print(f"Step {step_count}: Position {state}, Reward {reward}")
        env.render(mode="pygame", fps=10, delay=0.5)
        step_count += 1
        if done and state == goal:
            print(f"Episode complete. Total reward: {reward}, Total steps: {step_count}")
            print(f"Path taken: {path}")
            break

    if not done:
        print("Agent did not reach goal within step limit. Check training.")

if __name__ == "__main__":
    main()
