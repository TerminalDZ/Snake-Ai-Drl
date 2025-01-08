# Multi-Agent Snake Game with Deep Learning

## Overview
This project implements the classic Snake game using artificial intelligence, where multiple agents are trained simultaneously using Reinforcement Learning and Deep Neural Networks.

## Main Components

### 1. Neural Network (SnakeNN)
- Deep neural network consisting of 4 linear layers
- Uses LeakyReLU activation function
- Input: 15 values representing game state
- Output: 3 values representing possible actions (continue, turn right, turn left)

### 2. Shared Memory (SharedMemory)
- Stores previous experiences of agents
- Tracks training data (scores, epsilon, losses, average rewards)
- Uses threading.Lock mechanism for synchronization between agents

### 3. Snake Game (SnakeGame)
- Implements core game logic
- Determines game state (snake position, food, dangers)
- Renders game using Pygame

### 4. Agent (SnakeAgent)
- Makes decisions based on game state
- Uses epsilon-greedy strategy to balance exploration and exploitation
- Learns by updating neural network using Q-learning algorithm

## How the System Works

### 1. Game State
The game state consists of 15 values:
- 3 values for dangers (front, right, left)
- 4 values for current snake direction
- 4 values for relative food position
- 4 values for distance to snake's tail

### 2. Reward System
- +10 base points for eating food
- Additional reward based on snake length
- Small reward for moving towards food
- Penalty (-10 or -20) for collision
- Penalty for moving in circles

### 3. Training Process
1. Initialize 4 agents playing simultaneously
2. Each agent:
   - Gets game state
   - Chooses action (using epsilon-greedy)
   - Executes action and receives reward
   - Stores experience in shared memory
3. Periodic neural network training using random samples from experiences

### 4. Optimization Techniques
- Using Gradient Clipping to prevent gradient explosion
- Learning rate scheduling for better convergence
- Epsilon decay for exploration-exploitation balance
- Large experience memory (9 million experiences)

## How to Run Training

1. Ensure required libraries are installed:
   ```bash
   pip install numpy torch pygame
   ```

2. Run the file:
   ```bash
   python multi.py
   ```

3. A window will appear showing:
   - 4 games running in parallel
   - Best score achieved
   - Total number of attempts

## Important Notes
- Training parameters can be modified such as:
  - Number of agents (num_snakes)
  - Batch size (batch_size)
  - Learning rate (learning_rate)
  - Epsilon decay rate (epsilon_decay)
- Training continues until:
  - Reaching specified number of episodes
  - Manually closing the window
  - Program interruption (Ctrl+C)
