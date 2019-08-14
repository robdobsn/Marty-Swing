import gym
import gym_martyswing
import time
import numpy as np

# Create the MartySwing
env = gym.make('MartySwing-v0')

# First action is to remain straight-legged
nextAction = 1

# Reset the Gym
observation = env.reset()

# Go through a number of swings
while(True):
    # This display the MartySwing in a window
    env.render()

    # Take the next action
    observation, reward, done, info = env.step(nextAction)

    # Select a new action based on the position in the swing cycle
    if (info["theta"] < np.radians(5) and info["theta"] > np.radians(-5)) and info["kickAngle"] < 0.01:
        # Kick if we are near the middle and haven't kicked already
        nextAction = 0
    elif (info["theta"] < np.radians(-29) or info["theta"] > np.radians(29)) and info["kickAngle"] > 0.01:
        # Straighten if we are near the top of our swing and not straight-legged already
        nextAction = 1

    # Check if we completed this test
    if done:
        print("Test after {} secs".format(info["t"]))
        break

    # A short delay to make the display look "normal"
    time.sleep(.1)

# Close the swing environment
env.close()
