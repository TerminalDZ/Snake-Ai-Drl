import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pygame
import time
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import threading
import queue

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRID_COLOR = (40, 40, 40)

# Snake colors for different instances
SNAKE_COLORS = [GREEN, BLUE, YELLOW, MAGENTA, CYAN]

# Initialize Pygame
pygame.init()

class DataLogger:
    def __init__(self, filename="training_data.csv"):
        self.filename = filename
        self.data_queue = queue.Queue()
        self.columns = ['episode', 'snake_id', 'score', 'steps', 'epsilon', 'loss']
        
        # Create file if doesn't exist
        if not os.path.exists(filename):
            pd.DataFrame(columns=self.columns).to_csv(filename, index=False)
            
        # Start logging thread
        self.running = True
        self.thread = threading.Thread(target=self._logging_thread)
        self.thread.start()
    
    def log(self, episode, snake_id, score, steps, epsilon, loss):
        self.data_queue.put({
            'episode': episode,
            'snake_id': snake_id,
            'score': score,
            'steps': steps,
            'epsilon': epsilon,
            'loss': loss
        })
    
    def _logging_thread(self):
        while self.running:
            try:
                data = self.data_queue.get(timeout=1)
                df = pd.DataFrame([data])
                df.to_csv(self.filename, mode='a', header=False, index=False)
            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False
        self.thread.join()

class Analytics:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        
    def update(self, data_file):
        try:
            df = pd.read_csv(data_file)
            
            # Clear surface
            self.surface.fill(WHITE)
            
            # Create matplotlib figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
            
            # Plot 1: Average score per episode
            df.groupby('episode')['score'].mean().plot(ax=ax1)
            ax1.set_title('Average Score')
            
            # Plot 2: Epsilon decay
            df.groupby('episode')['epsilon'].mean().plot(ax=ax2)
            ax2.set_title('Exploration Rate (Epsilon)')
            
            # Plot 3: Average steps per episode
            df.groupby('episode')['steps'].mean().plot(ax=ax3)
            ax3.set_title('Average Steps')
            
            # Plot 4: Loss over time
            df.groupby('episode')['loss'].mean().plot(ax=ax4)
            ax4.set_title('Training Loss')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert matplotlib figure to pygame surface
            fig.canvas.draw()
            img = fig.canvas.buffer_rgba()
            img = pygame.image.frombuffer(img, fig.canvas.get_width_height(), "RGBA")
            self.surface.blit(img, (0, 0))
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error updating analytics: {e}")

class SnakeNN(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super(SnakeNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

class SnakeGame:
    def __init__(self, width=20, height=20, cell_size=20, position=(0, 0), snake_id=0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.position = position
        self.snake_id = snake_id
        self.snake_color = SNAKE_COLORS[snake_id % len(SNAKE_COLORS)]
        
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        self.surface = pygame.Surface((self.screen_width, self.screen_height))
        
        self.reset()
        
    def reset(self):
        # Initialize snake in the middle of the grid
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # Place food in random location
        self.place_food()
        return self.get_state()
        
    def place_food(self):
        while True:
            self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if self.food not in self.snake:
                break
                
    def get_state(self):
        # Get snake head position
        head = self.snake[0]
        
        # Calculate distances to walls, food, and body
        state = []
        
        # Distance to walls
        state.append(head[0])  # Distance to left wall
        state.append(self.width - 1 - head[0])  # Distance to right wall
        state.append(head[1])  # Distance to top wall
        state.append(self.height - 1 - head[1])  # Distance to bottom wall
        
        # Distance to food
        state.append(self.food[0] - head[0])  # X distance to food
        state.append(self.food[1] - head[1])  # Y distance to food
        
        # Danger straight ahead
        next_pos = (head[0] + self.direction[0], head[1] + self.direction[1])
        state.append(1.0 if self.is_collision(next_pos) else 0.0)
        
        # Danger to right
        right_direction = (self.direction[1], -self.direction[0])
        right_pos = (head[0] + right_direction[0], head[1] + right_direction[1])
        state.append(1.0 if self.is_collision(right_pos) else 0.0)
        
        # Danger to left
        left_direction = (-self.direction[1], self.direction[0])
        left_pos = (head[0] + left_direction[0], head[1] + left_direction[1])
        state.append(1.0 if self.is_collision(left_pos) else 0.0)
        
        # Current direction
        state.append(self.direction[0])
        state.append(self.direction[1])
        
        return np.array(state)
        
    def is_collision(self, pos):
        # Check if position is outside boundaries
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return True
        # Check if position is on snake body
        if pos in self.snake[1:]:
            return True
        return False
        
    def draw(self):
        self.surface.fill(BLACK)
        
        # Draw grid
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.surface, GRID_COLOR, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.surface, GRID_COLOR, (0, y), (self.screen_width, y))
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = self.snake_color if i == 0 else BLUE
            rect = pygame.Rect(
                segment[0] * self.cell_size + 1,
                segment[1] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.surface, color, rect)
        
        # Draw food
        rect = pygame.Rect(
            self.food[0] * self.cell_size + 1,
            self.food[1] * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.surface, RED, rect)
        
        # Draw score
        font = pygame.font.Font(None, 24)
        score_text = font.render(f'Snake {self.snake_id + 1}: {self.score}', True, WHITE)
        self.surface.blit(score_text, (10, 10))

    def step(self, action):
        self.steps += 1
        
        # Convert action (0: straight, 1: right, 2: left) to new direction
        if action == 1:  # Turn right
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # Turn left
            self.direction = (-self.direction[1], self.direction[0])
        
        # Move snake
        new_head = (self.snake[0][0] + self.direction[0],
                   self.snake[0][1] + self.direction[1])
        
        # Check if game is over
        reward = 0
        if self.is_collision(new_head):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
            reward = -0.1  # Small negative reward to encourage finding food quickly
        
        return self.get_state(), reward, False

class SnakeAgent:
    def __init__(self, state_size=11, action_size=3, memory_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = SnakeNN(state_size, 256, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Load and learn from historical data if available
        self.load_historical_data()

    def load_historical_data(self, filename="training_data.csv"):
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            successful_games = df[df['score'] > 0]  # Focus on games where snake scored points
            
            if not successful_games.empty:
                # Process successful games data
                for _, game in successful_games.iterrows():
                    # Create synthetic experiences from successful games
                    state = np.random.uniform(0, 1, self.state_size)  # Approximate state
                    next_state = np.random.uniform(0, 1, self.state_size)
                    action = random.randrange(self.action_size)
                    reward = game['score']
                    done = True
                    
                    # Add to memory
                    self.remember(state, action, reward, next_state, done)
                
                # Initial training on historical data
                if len(self.memory) >= 32:
                    self.replay(32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        
        # Mix current experiences with historical ones
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([transition[0] for transition in minibatch]))
        actions = torch.LongTensor(np.array([transition[1] for transition in minibatch]))
        rewards = torch.FloatTensor(np.array([transition[2] for transition in minibatch]))
        next_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch]))
        dones = torch.FloatTensor(np.array([transition[4] for transition in minibatch]))

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update model
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

class ParallelTrainer:
    def __init__(self, num_snakes=4):
        self.num_snakes = num_snakes
        
        # Calculate window layout
        self.game_size = 400
        window_width = self.game_size * 2
        window_height = self.game_size * 2
        self.analytics_width = 800
        
        # Create main window
        self.screen = pygame.display.set_mode((window_width + self.analytics_width, window_height))
        pygame.display.set_caption('Parallel Snake AI Training')
        
        # Create games and agents
        self.games = []
        self.agents = []
        positions = [(0, 0), (self.game_size, 0), (0, self.game_size), (self.game_size, self.game_size)]
        
        for i in range(num_snakes):
            game = SnakeGame(width=20, height=20, cell_size=20, position=positions[i], snake_id=i)
            agent = SnakeAgent()
            self.games.append(game)
            self.agents.append(agent)
        
        # Create analytics display
        self.analytics = Analytics(self.analytics_width, window_height)
        
        # Create data logger
        self.logger = DataLogger()

    def train(self, episodes=1000, batch_size=32):
        clock = pygame.time.Clock()
        episode = 0
        
        while episode < episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            self.screen.fill(BLACK)
            
            # Draw dividing lines
            # Vertical line between game pairs
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (self.game_size, 0), 
                           (self.game_size, self.game_size * 2), 3)
            # Horizontal line between game pairs
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (0, self.game_size), 
                           (self.game_size * 2, self.game_size), 3)
            # Vertical line between games and analytics
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (self.game_size * 2, 0), 
                           (self.game_size * 2, self.game_size * 2), 3)
            
            # Update and draw games
            for game in self.games:
                game.draw()
                self.screen.blit(game.surface, game.position)
            
            # Update and draw analytics
            self.analytics.update("training_data.csv")
            self.screen.blit(self.analytics.surface, (self.game_size * 2, 0))
            
            # Process each snake
            for i, (game, agent) in enumerate(zip(self.games, self.agents)):
                if not game.game_over:
                    # Get action
                    action = agent.act(game.get_state())
                    next_state, reward, done = game.step(action)
                    
                    # Store experience
                    agent.remember(game.get_state(), action, reward, next_state, done)
                    
                    # Train
                    if len(agent.memory) > batch_size:
                        loss = agent.replay(batch_size)
                        self.logger.log(episode, i, game.score, game.steps, agent.epsilon, loss)
                    
                    if done:
                        game.reset()
            
            pygame.display.flip()
            clock.tick(60)
            
            if episode % 10 == 0:
                print(f"Episode: {episode}")
                scores = [game.score for game in self.games]
                print(f"Scores: {scores}")
                epsilons = [agent.epsilon for agent in self.agents]
                print(f"Epsilons: {epsilons}")
            
            episode += 1
        
        self.logger.stop()
        pygame.quit()

if __name__ == "__main__":
    trainer = ParallelTrainer(num_snakes=4)
    trainer.train(episodes=1000)