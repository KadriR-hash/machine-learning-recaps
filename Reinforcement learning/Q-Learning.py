import gym  # all you have to do to import and use open ai gym! #Developed by OpenAI
import numpy as np
import time
import matplotlib.pyplot as plt

# EXAMPLE :
env = gym.make('FrozenLake-v1')  # we are going to use the FrozenLake enviornment
print(env.observation_space.n)  # get number of states
print(env.action_space.n)  # get number of actions
env.reset()  # reset enviornment to default state

action = env.action_space.sample()  # get a random action
new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action
env.render()  # render the GUI for the enviornment

# -------------------------------------------------------------------------------------------------------------------

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500  # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment
LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96
RENDER = True  # if you want to see training set to true

epsilon = 0.9

rewards = []
for episode in range(EPISODES):

    state = env.reset()
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE * (
                reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards) / len(rewards)}:")


# and now we can see our Q values!

# we can plot the training progress and see how the agent improved
def get_average(values):
    return sum(values) / len(values)


avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i + 100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()
