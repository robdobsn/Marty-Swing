import gym
import gym_martyswing
import time
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('martyswing-v0')

tim = []
ke = []
pe = []
th = []

nextAction = 0
for i_episode in range(1):
    observation = env.reset()
    for t in range(300):
        env.render()
        # print(observation)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(nextAction)
        tim.append(info["t"])
        pe.append(info["PE"])
        ke.append(info["KE"]) 
        th.append(info["theta"])
        # print(np.degrees(info["theta"]), info["PE"])
        if (info["theta"] < np.radians(5) and info["theta"] > np.radians(-5)) and info["kickAngle"] < 0.01:
            nextAction = 1
        elif (info["theta"] < np.radians(-26) or info["theta"] > np.radians(26)) and info["kickAngle"] > 0.01:
            nextAction = 2
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        time.sleep(.1)
            
env.close()

plt.plot(tim, ke, 'r')
plt.plot(tim, pe, 'b')
plt.plot(tim, th, 'g')
plt.show()
