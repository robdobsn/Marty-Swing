import gym
import gym_martyswing
import time
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MartySwing-v0')
env.thetaInit = np.radians(40)
env.dt = 0.005

tim = []
ke = []
pe = []
# th = []
# rewardTotal = []

nextAction = 1
for i_episode in range(1):
    observation = env.reset()
    rewardSum = 0
    while(True):
        env.render()
        # print(observation)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(nextAction)
        tim.append(info["t"])
        pe.append(info["PE"])
        ke.append(info["KE"]) 
        # th.append(info["theta"])
        # rewardSum += reward / 50
        # rewardTotal.append(rewardSum)
        # print(np.degrees(info["theta"]), info["PE"])
        # if (info["theta"] < np.radians(5) and info["theta"] > np.radians(-5)) and info["kickAngle"] < 0.01:
        #     nextAction = 0
        # elif (info["theta"] < np.radians(-29) or info["theta"] > np.radians(29)) and info["kickAngle"] > 0.01:
        #     nextAction = 1
        if info["t"] > 5 or done:
            print("Episode finished after {} secs".format(info["t"]))
            break
            
env.close()

plt.plot(tim, ke, 'r')
plt.plot(tim, pe, 'b')
# plt.plot(tim, th, 'g')
# plt.plot(tim, rewardTotal, 'p')
plt.show()
