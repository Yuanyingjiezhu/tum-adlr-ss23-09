import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical


# Define the environment parameters
TIME_STEPS = 120
BASIS_POINTS = 50
STATE_HISTORY = 3  # Number of previous time steps to consider in the state
STATE_DIM = BASIS_POINTS * BASIS_POINTS * STATE_HISTORY + 6  # SDF history + agent position (x, y) + start point (x0, y0) + goal point (xt, yt)
ACTION_DIM = 4  # Four possible actions: move up, down, left, or right


def calculate_reward(basis_points, sdf_values, agent_position):
    # Define the parameters for the reward function
    reach_goal_reward = 10  # Reward for reaching the goal
    collision_penalty = -5  # Penalty for colliding with obstacles
    obstacle_distance_factor = 0.5  # Penalty based on the distance to the nearest obstacle
    goal_distance_factor = -1  # Penalty based on the distance to the goal

    # Calculate the Euclidean distance between agent and each basis point
    distances = np.linalg.norm(np.array(basis_points) - agent_position, axis=1)

    # Find the index of the nearest basis point
    nearest_index = np.argmin(distances)
    nearest_distance = distances[nearest_index]
    nearest_sdf = sdf_values[nearest_index]

    # Check if the nearest basis point indicates a collision
    if nearest_sdf < 0:
        return collision_penalty

    # Calculate the Euclidean distance from the agent's position to the goal
    distance_to_goal = np.linalg.norm(agent_position - goal_point)

    # Compute the reward based on the distance to the nearest obstacle and the distance to the goal
    obstacle_reward = obstacle_distance_factor * nearest_distance
    goal_reward = goal_distance_factor * distance_to_goal

    # Combine the rewards
    reward = obstacle_reward + goal_reward

    # Add a positive reward if the agent reaches the goal
    if np.array_equal(agent_position, goal_point):
        reward += reach_goal_reward

    return reward


def build_base_points(num_points):
    base_points = []
    # Define the number of points and the area of the square
    # num_base_points = num_points * num_points
    # Generate the points
    x_coords = np.linspace(-1, 1, num_points)
    y_coords = np.linspace(-1, 1, num_points)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Flatten the meshgrid to obtain the individual coordinates
    x_samples = X.flatten()
    y_samples = Y.flatten()
    for i in range(len(x_samples)):
        point = (x_samples[i], y_samples[i])
        base_points.append(point)
    return base_points


# Define the RL agent's policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 64)
        self.fc2 = nn.Linear(64, ACTION_DIM)

    def forward(self, x):
        x = nn.LeakyReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action_probs(self, state):
        logits = self.forward(state)
        action_probs = torch.softmax(logits, dim=-1)
        print("Logits:", logits)
        print("Action Probs:", action_probs)
        print("NaN in action_probs:", torch.isnan(action_probs).any())
        return action_probs


# Define the trajectory planning RL agent
class RLAgent:
    def __init__(self):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.00001)
        self.gamma = 0.99  # Discount factor

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.policy.get_action_probs(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    # TPO,
    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in rewards[::-1]:
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
        print("Loss:", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Initialize the RL agent
agent = RLAgent()

# Define the sequential data
data = np.loadtxt('data/sdf_dataset_1')
sequential_data = data[:120, :]
agent_position = np.array([0.0, 0.0])  # Agent's initial position
start_point = np.array([0.0, 0.0])  # Start point coordinates
goal_point = np.array([1.0, 1.0])  # Goal point coordinates


# Training loop
for t in range(TIME_STEPS):
    if t >= (STATE_HISTORY - 1):
        state = sequential_data[t - STATE_HISTORY + 1:t + 1].flatten()
        state = np.concatenate((state, agent_position, start_point, goal_point))
        print(len(state))
        log_probs = []
        rewards = []

        trajectory = [agent_position]
        for _ in range(10):  # Generate 10 actions for each time step
            action, log_prob = agent.select_action(state)
            # Update agent's position based on the selected action
            if action == 0:  # Move up
                agent_position[1] += 0.001
            elif action == 1:  # Move down
                agent_position[1] -= 0.001
            elif action == 2:  # Move left
                agent_position[0] -= 0.001
            elif action == 3:  # Move right
                agent_position[0] += 0.001
            trajectory.append(agent_position)

            # Apply the action to the environment and observe the next state and reward
            obstacle_sdf = sequential_data[t]  # Get the SDF values at the current time step
            next_state = np.concatenate((state[:BASIS_POINTS*BASIS_POINTS*STATE_HISTORY], agent_position, start_point, goal_point))
            reward = calculate_reward(build_base_points(BASIS_POINTS), obstacle_sdf, agent_position)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            # Update the policy after generating a sequence of actions
            agent.update_policy(rewards, log_probs)

        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
        plt.scatter(start_point[0], start_point[1], c='g', label='Start')
        plt.scatter(goal_point[0], goal_point[1], c='r', label='Goal')
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c='b', label='Trajectory')
        plt.title("Episode: {} - Step: {}".format(t, t+1))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
