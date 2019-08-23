# Exhaustive approach to exploring the problem of Marty swinging
# -*- coding: utf-8 -*-

import gym
import gym_martyswing
import numpy as np
import time, math, random
import matplotlib.pyplot as plt
import keyboard
import itertools

# Create the MartySwing environment
env = gym.make('MartySwing-v0')

# Discrete actions
numActions = env.action_space.n # (kick, straight)
actionNames = ["Straight", "Kick"]
ACTION_STRAIGHT = 0
ACTION_KICK = 1
# Bounds for each state
stateBounds = (env.observation_space.low, env.observation_space.high)
# Discrete bounds for observation
xAccNumBins = 9
xAccBinBounds = np.linspace(stateBounds[0], stateBounds[1], xAccNumBins)
xAccBinBounds = xAccBinBounds.flatten()
# Used bin start and count
usedBinStart = 3
usedBinCount = 7

# Permutations of actions
# This is a list of tables which represent the action to perform in a specific state
# So each table is like a Q-Table except that it contains the action number for the best action
permutationsTable = [perm for perm in itertools.product(range(numActions), repeat=usedBinCount)]

# Goal and debug settings
EPISODES_PER_PERMUTATION = 1
TIME_MAX = 1000
STREAK_LEN_WHEN_DONE = 100
REWARD_SUM_GOAL = 20
LOG_DEBUG = True
LOG_DEBUG_FILE = "testruns/martySwingExhaustiveLog.txt"

# Main
def learnToSwing():

    # Track progress
    streaksNum = 0
    rewardTotal = []

    # Debug
    logDebugFile = None
    if LOG_DEBUG:
        debugActPrev = -1
        try:
            logDebugFile = open(LOG_DEBUG_FILE, "w+")
        except:
            print(f"Cannot write to log file {LOG_DEBUG_FILE}")
            exit(0)

    # Iterate permutations
    for permIdx in range(len(permutationsTable)):

        # Iterate episodes
        for episode in range(EPISODES_PER_PERMUTATION):

            # Reset the environment
            observation = env.reset()
            rewardSum = 0

            # Initial state
            statePrev = getObservationBinned(observation[0], xAccBinBounds)

            # Run the experiment over time steps
            for t in range(TIME_MAX):

                # Check if we're close to done for debugging
                if streaksNum > STREAK_LEN_WHEN_DONE - 1:
                    env.render()
                    time.sleep(0.1)

                # Select an action
                action = permutationsTable[permIdx][statePrev - usedBinStart]

                # Execute the action
                observation, reward, done, _ = env.step(action)
                state = getObservationBinned(observation[0], xAccBinBounds)

                # Get the new state
                rewardInState = reward
                done = False
                while not done:
                    state = getObservationBinned(observation[0], xAccBinBounds)
                    if state != statePrev:
                        break
                    observation, reward, done, _ = env.step(action)
                    rewardInState += reward

                # Sum rewards
                rewardSum += rewardInState

                # Setting up for the next iteration
                statePrev = state

                # Print data
                if logDebugFile is not None:
                    if debugActPrev != action:
                        logDebugFile.write(f"{actionNames[action]} --- Perm {permutationsTable[permIdx]} Ep {episode} t {t} Act {action} xAccBin {state} rew {rewardInState} Streaks {streaksNum}\n")
                        debugActPrev = action
                    # ky = keyboard.read_key()
                    # if ky == 'esc':
                    #     exit(0)

                if done:
                    rewardTotal.append(rewardSum)
                    logStr = f"{permIdx} -> {permutationsTable[permIdx]} rewardSum {rewardSum} time steps {t}"
                    if logDebugFile:
                        logDebugFile.write("....." + logStr + "\n")
                    print(logStr)
                    if (rewardSum >= REWARD_SUM_GOAL):
                        streaksNum += 1
                    else:
                        streaksNum = 0
                    break

            # It's considered done when it's solved over 100 times consecutively
            if streaksNum > STREAK_LEN_WHEN_DONE:
                break

    # Close debug log
    if logDebugFile:
        logDebugFile.close()

    plt.plot(rewardTotal, 'p')
    plt.show()

def getObservationBinned(val, bins):
    return np.digitize(val, bins)

if __name__ == "__main__":
    learnToSwing()
