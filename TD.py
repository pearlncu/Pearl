import numpy as np

class FrozenLake:
    def __init__(self, size=4, start_state=0, goal_state=None):
        self.size = size
        self.start_state = start_state
        self.goal_state = goal_state if goal_state is not None else size * size - 1
        self.current_state = start_state
        self.done = False

    def reset(self):
        self.current_state = self.start_state
        self.done = False
        return self.current_state

    def step(self, action):
        if self.done:
            raise ValueError("Episode has already terminated. Please reset the environment.")

        if action not in [0, 1, 2, 3]:  # 0: left, 1: down, 2: right, 3: up
            raise ValueError("Invalid action. Use 0, 1, 2, or 3.")

        row, col = divmod(self.current_state, self.size)

        # Transition dynamics
        if action == 0:  # left
            col = max(0, col - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # right
            col = min(self.size - 1, col + 1)
        elif action == 3:  # up
            row = max(0, row - 1)

        new_state = row * self.size + col

        # Reward and done flag
        if new_state == self.goal_state:
            reward = 1.0
            self.done = True
        else:
            reward = 0.0

        self.current_state = new_state

        return new_state, reward, self.done

def td_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99):
    # Initialize the value function
    V = np.zeros(env.size * env.size)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Choose an action using an epsilon-greedy policy
            action = epsilon_greedy_policy(env, V, state, epsilon=0.1)

            # Take the chosen action and observe the next state and reward
            next_state, reward, done = env.step(action)

            # Update the value function using TD update rule
            td_error = reward + gamma * V[next_state] - V[state]
            V[state] = V[state] + alpha * td_error

            state = next_state

    return V

def epsilon_greedy_policy(env, V, state, epsilon):
    # Epsilon-greedy policy: choose a random action with probability epsilon,
    # otherwise choose the action with the highest estimated value
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2, 3])
    else:
        return np.argmax([V[state] for _ in range(4)])

if __name__ == "__main__":
    # Create Frozen Lake environment
    env = FrozenLake(size=4, start_state=0, goal_state=15)

    # Perform TD learning
    learned_values = td_learning(env)

    # Print the learned values
    print("Learned Values:")
    print(learned_values.reshape(4, 4))
