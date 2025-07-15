
# 🚗 Autonomous Car in Grid World 🧠

A simulation project where an AI-powered autonomous car learns to navigate a grid environment with obstacles using **Q-Learning**, a fundamental Reinforcement Learning algorithm.

---

## 📌 Project Description

This project demonstrates how a self-driving car agent learns to reach a target in a 2D grid world while avoiding obstacles and minimizing steps. The environment is a matrix, and the car uses trial and error (reinforcement learning) to update its Q-table and learn optimal paths over time.

---

## 🎯 Features

- Grid world simulation with walls, start, and goal
- Reinforcement Learning with **Q-Learning**
- Customizable environment size and layout
- Reward-based feedback mechanism
- Optional: Visual representation of car movement (using `pygame` or `matplotlib`)
- Tracks agent's learning progress across episodes

---

## 🧠 AI Concepts Used

- Reinforcement Learning
- Q-Table: State-Action Value Matrix
- Exploration vs Exploitation (ε-greedy strategy)
- State, Action, Reward modeling
- Path optimization and convergence analysis

---

## 🗂️ Project Structure

```
autonomous-car-grid/
├── src/
│   ├── environment.py          # Grid environment logic
│   ├── agent.py                # Q-learning implementation
│   ├── main.py                 # Simulation runner and training loop
│   ├── visualizer.py           # Optional real-time animation
├── assets/
│   ├── grid_config.json        # Predefined environment
│   ├── car_icon.png            # Car image (if GUI used)
│   ├── q_table.npy             # Saved Q-table
├── tests/
│   ├── test_environment.py     # Unit tests
│   └── test_agent.py
├── docs/
│   ├── design_doc.pdf
│   └── test_plan.pdf
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 🧰 Prerequisites
- Python 3.7+
- pip

### 📦 Installation

```bash
git clone https://github.com/yourusername/autonomous-car-grid.git
cd autonomous-car-grid
pip install -r requirements.txt
```

### ▶️ Run Training

```bash
python src/main.py
```

### 📈 Visualize Training (optional)

```bash
python src/visualizer.py
```

---

## ⚙️ Configuration

You can modify the environment (grid size, start, goal, obstacles) by editing the `grid_config.json` inside the `assets/` folder.

---

## 📊 Sample Results

- Average reward across 1000 episodes
- Q-table convergence graphs
- Steps to goal over time

---

## 📚 References

- [Reinforcement Learning – Sutton & Barto](http://incompleteideas.net/book/the-book.html)
- [Sentdex Q-Learning Series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)
- [OpenAI Gym (for future expansion)](https://www.gymlibrary.dev/)

---

## 🧑‍💻 Author

**Jayaprakash A**  
📧 [jayaprakashoffic@gmail.com](mailto:jayaprakashoffic@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/jayaprakashoffic)

---

## ✅ License

This project is open-source and available under the [MIT License](LICENSE).
