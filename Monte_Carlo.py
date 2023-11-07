import numpy as np

# Define the simple environment
class Environment:
    def __init__(self):
        self.num_states = 5
        self.num_actions = 2
        self.transitions = {
            0: [1, 2],
            1: [0, 3],
            2: [0, 3],
            3: [1, 4],
            4: [2, 4]
        }
        self.rewards = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}

    def step(self, state, action):
        next_state = np.random.choice(self.transitions[state])
        reward = self.rewards[next_state]
        return next_state, reward


# First-visit Monte Carlo algorithm
def first_visit_monte_carlo(env, num_episodes, gamma=0.9):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    for _ in range(num_episodes):
        episode = []
        state = 0  # starting state
        while True:
            action = np.random.randint(env.num_actions)  # choose a random action
            next_state, reward = env.step(state, action)
            episode.append((state, reward))
            if next_state == 4:  # terminal state
                break
            state = next_state

        states_in_episode = set([x[0] for x in episode])
        for state in states_in_episode:
            first_occurrence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            G = sum([x[1] * (gamma ** i) for i, x in enumerate(episode[first_occurrence_idx:])])
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

    return V


if __name__ == '__main__':
    from collections import defaultdict

    env = Environment()
    num_episodes = 10
    V = first_visit_monte_carlo(env, num_episodes)
    print("Value function:")
    for state, value in V.items():
        print(f"State {state}: {value:.2f}")
