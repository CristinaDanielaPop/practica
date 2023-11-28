import gym
import numpy as np

# presupunem un mediu simplu
class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # inainte, inapoi, stanga, dreapta
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)

    def reset(self):
        # resetare mediu
        return np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)

    def step(self, action):
        # simularea miscarii robotului pe baza actiunii
        # returneaza stare noua, recompensa si indeplinirea sarcinii
        state = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8) 
        reward = 1.0  
        done = False  
        return state, reward, done, {}

# Q-learning - parametri
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1
num_episodes = 1000

# initializare Q-table
env = RobotEnv()
state_space_size = np.prod(np.array((height, width, channels)))
action_space_size = 4
q_table = np.zeros((state_space_size, action_space_size))

# Q-learning - algoritm
for episode in range(num_episodes):
    state = env.reset()

    total_reward = 0

    while True:
        # alegerea unei actiuni
        if np.random.rand() < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(q_table[state])

        # ia actiuna aleasa
        next_state, reward, done, _ = env.step(action)

        # actualizare Q-value folosind regula de actualizare Q-learning 
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
            learning_rate * (reward + discount_factor * np.max(q_table[next_state]))

        total_reward += reward
        state = next_state

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# inchide mediul (daca este aplicabil)
env.close()