# 🐍 Snake AI - Deep Reinforcement Learning

A sophisticated implementation of the classic Snake game powered by Deep Reinforcement Learning, featuring parallel training capabilities and real-time analytics.

## 🌟 Features

- **Multi-Agent Training**: Train up to 4 snake agents simultaneously
- **Real-Time Analytics**: Monitor training progress with live graphs and statistics
- **Deep Neural Network**: Implements a 3-layer neural network for decision making
- **Advanced Visualization**: Pygame-based graphical interface with color-coded snakes
- **Data Logging**: Comprehensive training data logging and analysis
- **Adaptive Learning**: Epsilon-greedy strategy with decay for exploration/exploitation balance

## 🛠️ Technologies Used

- Python 3.x
- PyTorch
- Pygame
- NumPy
- Pandas
- Matplotlib

## 📋 Prerequisites

```bash
pip install torch numpy pygame pandas matplotlib
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/terminalDZ/snake-ai-drl.git
cd snake-ai-drl
```

2. Run the single snake training:
```bash
python singal.py
```

3. Run the parallel training with multiple agents  and analytics:
```bash
python game.py
```
Or run the parallel training with multiple agents:
```bash
python multi.py
```

## 🎮 Game Environment

- Grid Size: 20x20
- Actions: 3 (Turn Left, Go Straight, Turn Right)
- State Space: 11 dimensions including:
  - Danger detection in 3 directions
  - Current direction
  - Food location relative to snake
  - Current score and steps

## 📊 Training Parameters

- Batch Size: 32
- Memory Size: 100,000 steps
- Learning Rate: 0.001
- Discount Factor (γ): 0.95
- Initial Exploration Rate (ε): 1.0
- Minimum Exploration Rate: 0.01
- Exploration Decay: 0.995

## 📈 Analytics

The training interface provides real-time visualization of:
- Average Score
- Exploration Rate (Epsilon)
- Average Steps per Episode
- Training Loss

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 👤 Author

**Idriss Boukmouche**

- GitHub: [@terminalDZ](https://github.com/terminalDZ)

