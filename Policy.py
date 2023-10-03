import numpy as np

# Define the MDP (Markov Decision Process)
num_states = 3
num_actions = 2

# Define the transition probabilities (P), rewards (R), and initial policy (pi)
# These are example values; you should replace them with your specific MDP details.
P = np.zeros((num_states, num_actions, num_states))  # Transition probabilities
R = np.zeros((num_states, num_actions, num_states))  # Rewards
pi = np.zeros((num_states, num_actions))  # Initial policy

# Define the discount factor
gamma = 0.9

# Policy Iteration Algorithm
def policy_evaluation(pi, P, R, gamma):
    num_states, num_actions = pi.shape
    V = np.zeros(num_states)  # Initialize value function
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            action = int(pi[s].argmax())
            V[s] = sum([P[s, action, s1] * (R[s, action, s1] + gamma * V[s1]) for s1 in range(num_states)])
            delta = max(delta, abs(v - V[s]))
        
        if delta < 1e-6:
            break
    
    return V

def policy_improvement(pi, P, R, gamma):
    num_states, num_actions = pi.shape
    policy_stable = True
    
    for s in range(num_states):
        old_action = int(pi[s].argmax())
        
        # Compute Q-values for each action
        Q = np.zeros(num_actions)
        for a in range(num_actions):
            Q[a] = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(num_states)])
        
        # Update policy to choose the action with the highest Q-value
        best_action = np.argmax(Q)
        pi[s] = np.eye(num_actions)[best_action]
        
        if best_action != old_action:
            policy_stable = False
    
    return pi, policy_stable

# Policy Iteration
while True:
    V = policy_evaluation(pi, P, R, gamma)
    pi, policy_stable = policy_improvement(pi, P, R, gamma)
    
    if policy_stable:
        break

# Print the final policy and value function
print("Optimal Policy:")
print(pi)
print("Optimal Value Function:")
print(V)