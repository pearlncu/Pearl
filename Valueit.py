import numpy as np

# Define the environment
env = np.array([[-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]])

# Define the reward table
rewards = np.array([[0, 0, 0, 100],
                    [0, -1, 0, -1],
                    [0, 0, 0, -1],
                    [0, 0, 0, 0]])

# Set hyperparameters
gamma = 0.8  # Discount factor
epsilon = 0.001  # Convergence threshold

# Initialize the value table
values = np.zeros((4, 4))

# Define the available actions
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Implement the value iteration algorithm
while True:
    delta = 0
    new_values = np.copy(values)
    for i in range(4):
        for j in range(4):
            if (i, j) == (0, 3):
                continue
            value_list = []
            for action in actions:
                next_i, next_j = i + action[0], j + action[1]
                if 0 <= next_i < 4 and 0 <= next_j < 4:
                    value_list.append(rewards[next_i, next_j] + gamma * values[next_i, next_j])
                else:
                    value_list.append(rewards[i, j])
            new_values[i, j] = max(value_list)
            delta = max(delta, abs(new_values[i, j] - values[i, j]))
    values = new_values
    if delta < epsilon:
        break

# Print the final values
print("Optimal Values:")
print(values)