import setup_path
import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import os  # For checking if the weights file exists

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the last convolutional layer
        self.conv_output_size = self._get_conv_output_size((2, 84, 84))
        
        self.fc1 = nn.Linear(self.conv_output_size + state_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def _get_conv_output_size(self, shape):
        input = torch.rand(1, *shape)
        output = self.conv3(self.conv2(self.conv1(input)))
        return int(np.prod(output.size()))

    def forward(self, image, state):
        x = torch.relu(self.conv1(image))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        state = state.float()
        x = torch.cat((x, state), dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DroneAgent:
    def __init__(self, state_size, action_size, device, weights_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Load weights if a path is provided and file exists
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
            print(f"Loaded model weights from {weights_path}")

    def remember(self, state, image, action, reward, next_state, next_image, done):
        self.memory.append((state, image, action, reward, next_state, next_image, done))

    def act(self, state, image):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        image = torch.FloatTensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(image, state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, image, action, reward, next_state, next_image, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            image = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_image = torch.FloatTensor(next_image).unsqueeze(0).to(self.device)
            
            target = reward
            if not done:
                with torch.no_grad():
                    next_q_values = self.model(next_image, next_state)
                    max_next_q_value = torch.max(next_q_values).item()
                    target = reward + self.gamma * max_next_q_value
            
            q_values = self.model(image, state)
            target_q_values = q_values.clone()
            target_q_values[0][action] = target
            
            loss = nn.MSELoss()(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# AirSim interaction
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Define state and action space
state_size = 6  # x, y, z, roll, pitch, yaw
action_size = 7  # move forward, backward, left, right, up, down, hover

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = 'C:/Users/Kostas/AirSim/PythonClient/multirotor/weights/dqn_weights.pth'
agent = DroneAgent(state_size, action_size, device, weights_path=weights_path)

def get_image_with_depth():
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
    ])
    
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    depth = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)
    depth = depth.astype(np.float32)
    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    grayscale = cv2.resize(grayscale, (84, 84))
    depth = cv2.resize(depth, (84, 84))

    image = np.stack([grayscale, depth], axis=0)
    return image.astype(np.float32)

def check_collision():
    collision_info = client.simGetCollisionInfo()
    return collision_info.has_collided

def reset_drone():
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

def calculate_orientation_reward(drone_forward, target_direction):
    """Calculate how well the drone is facing the target."""
    dot_product = np.dot(drone_forward, target_direction)
    alignment_reward = (dot_product + 1) / 2  # Normalize to range [0, 1]
    return alignment_reward * 10  # Scale reward

# Training loop
episodes = 1000
max_negative_rewards = 5

for e in range(episodes):
    if e % 100 == 0:  # save weights every 100 episodes
        torch.save(agent.model.state_dict(), weights_path)
    reset_drone()

    drone_state = client.getMultirotorState()
    position = drone_state.kinematics_estimated.position.to_numpy_array()
    orientation = airsim.to_eularian_angles(drone_state.kinematics_estimated.orientation)
    state = np.concatenate([position, orientation])
    
    image = get_image_with_depth()
    
    target = np.array([5042, -17852, 13857], dtype=np.float32)
    previous_dist = np.linalg.norm(state[:3] - target)
    
    negative_reward_count = 0
    total_reward = 0
    
    for time in range(10000):
        action = agent.act(state, image)

        # Execute action based on the agent's decision
        if action == 0:
            client.moveByVelocityAsync(1, 0, 0, 1).join()
        elif action == 1:
            client.moveByVelocityAsync(-1, 0, 0, 1).join()
        elif action == 2:
            client.moveByVelocityAsync(0, 1, 0, 1).join()
        elif action == 3:
            client.moveByVelocityAsync(0, -1, 0, 1).join()
        elif action == 4:
            client.moveByVelocityAsync(0, 0, -1, 1).join()
        elif action == 5:
            client.moveByVelocityAsync(0, 0, 1, 1).join()
        elif action == 6:
            client.moveByVelocityAsync(0, 0, 0, 1).join()  # Hover action

        # Get new state and image
        drone_state = client.getMultirotorState()
        next_position = drone_state.kinematics_estimated.position.to_numpy_array()
        next_orientation = airsim.to_eularian_angles(drone_state.kinematics_estimated.orientation)
        next_state = np.concatenate([next_position, next_orientation])
        next_image = get_image_with_depth()

        # Calculate direction to the target and drone's forward direction
        target_direction = target - next_position
        target_direction /= np.linalg.norm(target_direction)  # Normalize
        drone_forward = np.array([
            np.cos(next_orientation[2]),  # yaw direction (forward in x-y plane)
            np.sin(next_orientation[2]), 
            0
        ])

        # Calculate rewards
        current_dist = np.linalg.norm(next_state[:3] - target)
        dist_reward = previous_dist - current_dist
        orientation_reward = calculate_orientation_reward(drone_forward, target_direction)
        reward = dist_reward * 10 + orientation_reward  # Combine distance and orientation rewards
        
        if check_collision():
            reward -= 100
            done = True
        else:
            done = False

        if action == 6 and current_dist < 2:
            reward += 10

        if reward < 0:
            negative_reward_count += 1
        else:
            negative_reward_count = 0

        if negative_reward_count >= max_negative_rewards:
            done = True

        if current_dist < 1:
            reward += 100
            done = True
        
        total_reward += reward
        agent.remember(state, image, action, reward, next_state, next_image, done)
        
        state = next_state
        image = next_image
        previous_dist = current_dist
        
        if done:
            print(f"Episode {e + 1}/{episodes}, Time: {time}, Total Reward: {total_reward}")
            break

    # Train the agent after each episode
    if len(agent.memory) > 32:
        agent.replay(32)

# End of training loop
client.armDisarm(False)
client.enableApiControl(False)
