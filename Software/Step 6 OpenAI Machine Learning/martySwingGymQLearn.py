# With inspiration from https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# -*- coding: utf-8 -*-

import gym
import gym_martyswing
import numpy as np
import time, math, random
import matplotlib.pyplot as plt

# Create the MartySwing environment
env = gym.make('MartySwing-v0')

# Discrete actions
numActions = env.action_space.n # (straight, kick)
actionNames = ["Straight", "Kick", ""]
ACTION_STRAIGHT = 0
ACTION_KICK = 1
# Bounds for each state
stateBounds = (env.observation_space.low, env.observation_space.high)
# Discrete bounds for observation
xAccNumBins = 9
xAccBinBounds = np.linspace(stateBounds[0], stateBounds[1], xAccNumBins-1)
xAccBinBounds = xAccBinBounds.flatten()
# Directions
numDirections = 2

# Q Table indexed by state-action pair
qTable = np.zeros((xAccNumBins * numDirections, numActions))

# Learning and exploration settings
EXPLORATION_RATE_MAX = 0.7
EXPLORATION_RATE_MIN = 0.01
EXPLORATION_RATE_DECAY_FACTOR = 10
LEARN_RATE_MAX = 1.0
LEARN_RATE_MIN = 0.1
LEARN_RATE_DECAY_FACTOR = 50
DISCOUNT_FACTOR = 0.8

# Goal and debug settings
EPISODE_MAX = 500
TIME_MAX = 1000
STREAK_LEN_WHEN_DONE = 100
REWARD_SUM_GOAL = 2500
LOG_DEBUG = True
LOG_DEBUG_FILE = "testruns/martySwingQLearnLog.txt"

# Debug
learnRateVals = []
exploreRateVals = []

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
        episodeRewardSum = 0

        # Initial state
        statePrev = getObservationBinned(observation[0], xAccBinBounds)
        state = statePrev

        # Run the experiment over time steps
        t = 0
        rewardInState = 0
        action = ACTION_STRAIGHT
        while True:

            # Check if we're close to done for debugging
            if streaksNum > STREAK_LEN_WHEN_DONE - 1:
                env.render()
                time.sleep(0.1)

            # Execute the action
            observation, reward, done, info = env.step(action)
            t += 1
            state = getObservationBinned(observation[0], xAccBinBounds)

            # Accumulate rewards in this state
            rewardInState += reward

            # Log data
            if logDebugFile is not None:
                logDebugFile.write(f"{actionNames[action]} --- Ep {episode} t {t} statePrev {statePrev} state {state} rew {rewardInState:.2f} {'[+]' if rewardInState > 0 else ('[~]' if rewardInState > -1 else '[-]')} PE {info['PE']:.2f} KE {info['KE']:.2f} TE {info['PE']+info['KE']:.2f} theta {info['theta']:.2f} thetaMax {info['thetaMax']:.2f} v {info['v']:.2f} explRate {explorationRate} learnRate {learningRate} Streaks {streaksNum} \n")

            # Check if there has been a change of state
            if state != statePrev:
                # Update the Q Table using the Bellman equation
                best_q = np.amax(qTable[state])
                qTable[statePrev, action] += learningRate*(rewardInState + DISCOUNT_FACTOR*(best_q) - qTable[statePrev, action])

                # Debug
                if logDebugFile is not None:
                    logDebugFile.write(dumpQTable(qTable))

                # Select a new action
                action = actionSelect(state, explorationRate)

                # Sum rewards in episode
                episodeRewardSum += rewardInState
                rewardInState = 0

            # Ready for next iteration
            statePrev = state

            # Check for episode done
            if done or t > TIME_MAX:
                rewardTotal.append(episodeRewardSum)
                logStr = f"Episode {episode} finished after {t} episodeRewardSum {episodeRewardSum:.2f} thetaMax {info['thetaMax']:.2f} learnRate {learningRate:.2f} exploreRate {explorationRate:.2f} streakLen {streaksNum}"
                if logDebugFile:
                    logDebugFile.write("....." + logStr + "\n")
                if episode % 100 == 0:
                    print(dumpQTable(qTable))
                    print(logStr)
                if (episodeRewardSum >= REWARD_SUM_GOAL):
                    streaksNum += 1
                else:
                    streaksNum = 0
                break

        # It's considered done when it's solved over N times consecutively
        if streaksNum > STREAK_LEN_WHEN_DONE:
            break

        # Update parameters
        learnRateVals.append(learningRate)
        exploreRateVals.append(explorationRate)
        explorationRate = getExplorationRate(episode)
        learningRate = getLearningRate(episode)

    # Close debug log
    if logDebugFile:
        logDebugFile.close()

    print(dumpQTable(qTable))
    plt.plot(rewardTotal, 'p')
    plt.show()
    plt.plot(learnRateVals, 'g')
    plt.plot(exploreRateVals, 'b')
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
    return max(EXPLORATION_RATE_MIN, EXPLORATION_RATE_MAX * (1.0 - math.log10(t/EXPLORATION_RATE_DECAY_FACTOR+1)))

def getLearningRate(t):
    # Learning rate is a log function reducing over time
    return max(LEARN_RATE_MIN, LEARN_RATE_MAX * (1.0 - math.log10(t/LEARN_RATE_DECAY_FACTOR+1)))

# Sensing direction (using a moving average)
obsList = []
obsSum = 0
obsWindowLen = 3
def getObservationBinned(val, bins):
    global obsList, obsSum
    # Smooth the observations
    obsSumPrev = obsSum
    if len(obsList) >= obsWindowLen:
        obsSum -= obsList[0]
    else:
        obsList = [val] * (obsWindowLen-1)
        obsSum = val * (obsWindowLen-1)
        obsSumPrev = obsSum + val
    obsList = obsList[-(obsWindowLen-1):]
    obsList.append(val)
    obsSum += val
    discreteVal = np.digitize(val, bins)
    if obsSum >= obsSumPrev:
        return discreteVal
    return xAccNumBins * 2 - 1 - discreteVal

def dumpQTable(qTable):
    dumpStr = ""
    for i, st in enumerate(qTable):
        for ac in st:
            dumpStr += f"{ac:0.4f}\t"
        if st[0] == st[1]:
            bestAct = 2
        else:
            bestAct = np.argmax(st)
        dumpStr += f"{'RL ' if i < len(qTable)/2 else 'LR '} {actionNames[bestAct]}\n"
    return dumpStr
            
if __name__ == "__main__":
    learnToSwing()
