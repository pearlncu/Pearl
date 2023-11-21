import gym
import numpy as np
import random
env = gym.make('FrozenLake-v1', is_slippery=False)
action_space_size=env.action_space.n
action_space_size
state_space_size=env.observation_space.n
state_space_size
qtable=np.zeros((state_space_size, action_space_size))
qtable
#hyperparameters
total_episodes=10000
learning_rate=0.2
max_steps=100
gamma=0.99

epsilon=1
max_epsilon=1
min_epsilon=.01
decay_rate=0.001


rewards=[]
for episode in range(total_episodes):
  state=env.reset()
  step=0
  done=False
  total_rewards=0

  for step in range(max_steps):
    if random.uniform(0,1)>epsilon:
      action=np.argmax(qtable[state,:]) #exploitation
    else:
      action=env.action_space.sample() # Exploration

    new_state, reward, done, info =env.step(action)
    max_new_state=np.max(qtable[new_state,:])
    qtable[state,action] = qtable[state,action] + learning_rate*(reward+gamma*max_new_state-qtable[state,action])

    total_rewards+= reward
    state= new_state

    if done:
      break

  epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
  rewards.append(total_rewards)

print ("Score:", str(sum(rewards)/total_episodes))
