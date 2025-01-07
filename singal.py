import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pygame
import time

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRID_COLOR = (40, 40, 40)

# Initialize Pygame
pygame.init()

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
    def __init__(self, width=20, height=20, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Initialize Pygame display
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Snake AI')
        
        self.reset()
        
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (1, 0)
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        return self._get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, self.width-1), 
                   random.randint(0, self.height-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        head = self.snake[0]
        danger = [False, False, False]
        
        # Check straight
        next_pos = (head[0] + self.direction[0], head[1] + self.direction[1])
        if (next_pos[0] < 0 or next_pos[0] >= self.width or
            next_pos[1] < 0 or next_pos[1] >= self.height or
            next_pos in self.snake):
            danger[0] = True
            
        # Check right
        right_dir = self._turn_right(self.direction)
        next_pos = (head[0] + right_dir[0], head[1] + right_dir[1])
        if (next_pos[0] < 0 or next_pos[0] >= self.width or
            next_pos[1] < 0 or next_pos[1] >= self.height or
            next_pos in self.snake):
            danger[1] = True
            
        # Check left
        left_dir = self._turn_left(self.direction)
        next_pos = (head[0] + left_dir[0], head[1] + left_dir[1])
        if (next_pos[0] < 0 or next_pos[0] >= self.width or
            next_pos[1] < 0 or next_pos[1] >= self.height or
            next_pos in self.snake):
            danger[2] = True
            
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)
        
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        return np.array([
            *danger,
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)
    
    def _turn_right(self, direction):
        return (-direction[1], direction[0])
    
    def _turn_left(self, direction):
        return (direction[1], -direction[0])
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw grid
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.screen_width, y))
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = GREEN if i == 0 else BLUE  # Head is green, body is blue
            rect = pygame.Rect(
                segment[0] * self.cell_size + 1,
                segment[1] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw food
        rect = pygame.Rect(
            self.food[0] * self.cell_size + 1,
            self.food[1] * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.screen, RED, rect)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
    
    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, True
        
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)
            
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            return self._get_state(), -10, True
            
        self.snake.insert(0, new_head)
        
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            reward = -0.1
            
        self.steps += 1
        
        if self.steps > 100 * len(self.snake):
            self.game_over = True
            return self._get_state(), reward, True
        
        self.draw()
        time.sleep(0.05)  # Add delay to make the game visible
            
        return self._get_state(), reward, False

class SnakeAgent:
    def __init__(self, state_size=11, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = SnakeNN(state_size, 256, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        
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
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.LongTensor([i[1] for i in minibatch])
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor([i[4] for i in minibatch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(episodes=1000, batch_size=32):
    env = SnakeGame()
    agent = SnakeAgent()
    scores = []
    
    try:
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.game_over:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                if next_state is None:  # Window was closed
                    return agent, scores
                    
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                    
            scores.append(env.score)
            
            if e % 10 == 0:  # Print more frequently
                print(f"Episode: {e}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        
    finally:
        pygame.quit()
        
    return agent, scores

if __name__ == "__main__":
    trained_agent, training_scores = train(episodes=1000)
    print("Training completed!")
    print(f"Final average score: {sum(training_scores[-100:])/100}")