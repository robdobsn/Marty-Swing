# With inspiration from https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# -*- coding: utf-8 -*-

import gym
import gym_martyswing
import numpy as np
import time, math, random
import matplotlib.pyplot as plt
import keyboard

# Create the MartySwing environment
env = gym.make('martyswing-v0')

# Discrete actions
numActions = env.action_space.n # (nothing, kick, unkick)
actionNames = ["Kick", "Straight"]
# Bounds for each state
stateBounds = (env.observation_space.low, env.observation_space.high)
# Discrete bounds for observation
xAccBinBounds = [stateBounds[0]*0.5, stateBounds[0]*0.1, stateBounds[1]*0.1, stateBounds[1]*0.5]
# Number of discrete states (bins)
numStateBins = (len(xAccBinBounds)+1,)

# Q Table indexed by state-action pair
qTable = np.zeros(numStateBins + (numActions,))

# Learning and exploration settings
EXPLORATION_RATE_MIN = 0.01
EXPLORATION_RATE_ROLLOFF = 25
LEARN_RATE_MIN = 0.1
LEARNING_RATE_ROLLOFF = 25
DISCOUNT_FACTOR = 0.9

# Goal and debug settings
EPISODE_MAX = 200
TIME_MAX = 1000
STREAK_LEN_WHEN_DONE = 100
REWARD_SUM_GOAL = 20
LOG_DEBUG = True
LOG_DEBUG_FILE = "testruns/martySwingQLearnLog.txt"

# Main
def learnToSwing():

    # Set the learning and explore rates initially
    learningRate = getLearningRate(0)
    explorationRate = getExplorationRate(0)

    # Track progress in learning
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

    # Iterate episodes
    for episode in range(EPISODE_MAX):

        # Reset the environment
        observation = env.reset()
        rewardSum = 0

        # Initial state
        statePrev = getObservationBinned(observation[0])

        # Run the experiment over time steps
        for t in range(TIME_MAX):

            # Check if we're close to done for debugging
            if streaksNum > STREAK_LEN_WHEN_DONE - 2:
                env.render()
                time.sleep(0.1)

            # Select an action
            action = actionSelect(statePrev, explorationRate)

            # Execute the action
            observation, reward, done, _ = env.step(action)

            # Get the new state
            rewardInState = reward
            while True:
                state = getObservationBinned(observation[0])
                if state != statePrev:
                    break
                observation, reward, done, _ = env.step(action)
                rewardInState += reward

            # Sum rewards
            rewardSum += rewardInState

            # Update the Q Table using the Bellman equation
            best_q = np.amax(qTable[state])
            qTable[statePrev + (action,)] += learningRate*(reward + DISCOUNT_FACTOR*(best_q) - qTable[statePrev + (action,)])

            # Setting up for the next iteration
            statePrev = state

            # Print data
            if logDebugFile is not None:
                if debugActPrev != action:
                    logDebugFile.write(f"{actionNames[action]} --- Ep {episode} t {t} Act {action} xAccBin {state[0]} rew {reward} bestQ {best_q} explRate {explorationRate} learnRate {learningRate} Streaks {streaksNum}\n")
                    logDebugFile.write(dumpQTable(qTable))
                    debugActPrev = action
                # ky = keyboard.read_key()
                # if ky == 'esc':
                #     exit(0)

            if done:
                rewardTotal.append(rewardSum)
                logStr = "Episode %d finished after %f time steps rewardSum %f learnRate %f exploreRate %f" % (episode, t, rewardSum, learningRate, explorationRate)
                if logDebugFile:
                    logDebugFile.write("....." + logStr + "\n")
                print(logStr)
                print(dumpQTable(qTable))
                if (rewardSum >= REWARD_SUM_GOAL):
                    streaksNum += 1
                else:
                    streaksNum = 0
                break

        # It's considered done when it's solved over 100 times consecutively
        if streaksNum > STREAK_LEN_WHEN_DONE:
            break

        # Update parameters
        explorationRate = getExplorationRate(episode)
        learningRate = getLearningRate(episode)

    # Close debug log
    if logDebugFile:
        logDebugFile.close()

    print(dumpQTable(qTable))
    plt.plot(rewardTotal, 'p')
    plt.show()


def actionSelect(state, explorationRate):
    # The exploration rate determines the likelihood of taking a random
    # action vs the action with the best Q
    if random.random() < explorationRate:
        # Random action
        action = env.action_space.sample()
    else:
        # Action with best Q for current state
        action = np.argmax(qTable[state])
    return action

def getExplorationRate(t):
    # Exploration rate is a log function reducing over time
    return max(EXPLORATION_RATE_MIN, min(1.0, 1.0 - math.log10((t+1)/EXPLORATION_RATE_ROLLOFF)))

def getLearningRate(t):
    # Learning rate is a log function reducing over time
    return max(LEARN_RATE_MIN, min(1.0, 1.0 - math.log10((t+1)/LEARNING_RATE_ROLLOFF)))

def getObservationBinned(xAcc):
    for i in range(len(xAccBinBounds)):
        if xAcc < xAccBinBounds[i]:
            return (i,)
    return (len(xAccBinBounds),)

def dumpQTable(qTable):
    dumpStr = ""
    for st in qTable:
        for ac in st:
            dumpStr += f"{ac:0.4f}\t"
        bestAct = np.argmax(st)
        dumpStr += f"{actionNames[bestAct]}\n"
    return dumpStr
            
if __name__ == "__main__":
    learnToSwing()
