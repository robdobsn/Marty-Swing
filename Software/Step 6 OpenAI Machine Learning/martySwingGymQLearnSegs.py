# With inspiration from https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# -*- coding: utf-8 -*-

import gym
import gym_martyswing
import numpy as np
import time, math, random
import matplotlib.pyplot as plt
import matplotlib
import keyboard
import itertools

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

# Learning rate and exploration settings
EXPLORATION_RATE_MAX = 0.3
EXPLORATION_RATE_MIN = 0.1
EXPLORATION_RATE_DECAY_FACTOR = 5
LEARN_RATE_MAX = 0.3
LEARN_RATE_MIN = 0.1
LEARN_RATE_DECAY_FACTOR = 5
DISCOUNT_FACTOR = 0.99

# Goal and debug settings
EPISODE_MAX = 2000
TIME_MAX = 1000
STREAK_LEN_WHEN_DONE = 50
REWARD_SUM_GOAL = 1000
LOG_DEBUG = False
LOG_DEBUG_FILE = "testruns/martySwingQLearnSegLog.txt"
SHOW_RENDER = False
FIXED_ACTION = False
PERMUTE_ACTION = False

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

        # Setup for permutation
        if PERMUTE_ACTION:
            permuteTableSetup(episode)

        # Run the experiment over time steps
        t = 0
        rewardInState = 0
        action = ACTION_STRAIGHT
        while True:

            # Render the scene
            doRender(streaksNum)

            # Execute the action
            observation, reward, done, info = env.step(action)
            t += 1
            state = getObservationBinned(observation[0], xAccBinBounds)

            # Accumulate rewards in this state
            rewardInState += reward
           
            # Log data
            if logDebugFile is not None:
                # if debugActPrev != action:
                logDebugFile.write(f"{actionNames[action]} --- Ep {episode} t {t} statePrev {statePrev} state {state} rew {rewardInState:.2f} {'[+]' if rewardInState > 0 else ('[~]' if rewardInState > -1 else '[-]')} PE {info['PE']:.2f} KE {info['KE']:.2f} TE {info['PE']+info['KE']:.2f} theta {info['theta']:.2f} thetaMax {info['thetaMax']:.2f} v {info['v']:.2f} explRate {explorationRate} learnRate {learningRate} Streaks {streaksNum} \n")
                # debugActPrev = action
                # ky = keyboard.read_key()
                # if ky == 'esc':
                #     exit(0)

            # Check if there has been a change of state
            if state != statePrev:

                if not PERMUTE_ACTION:
                    # Update the Q Table using the Bellman equation
                    best_q = np.amax(qTable[state])
                    qTable[statePrev, action] += learningRate*(rewardInState + DISCOUNT_FACTOR*(best_q) - qTable[statePrev, action])

                # Debug
                if logDebugFile is not None:
                    logDebugFile.write(dumpQTable(qTable))

                # Select a new action
                if FIXED_ACTION:
                    action = actionSelectFix(episode, state, explorationRate)
                elif PERMUTE_ACTION:
                    action = actionSelectPermute(episode, state, explorationRate)                    
                else:
                    action = actionSelect(episode, state, explorationRate)

                # Sum rewards in episode
                episodeRewardSum += rewardInState
                rewardInState = 0

            # Ready for next iteration
            statePrev = state

            # Check for done
            if done or t > TIME_MAX or (PERMUTE_ACTION and permutesDone(episode)):
                rewardTotal.append(info["thetaMax"])
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

        # It's considered done when it's solved over 100 times consecutively
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


def actionSelect(episode, state, explorationRate):
    # if state >= xAccNumBins:
    #     return 0    
    # The exploration rate determines the likelihood of taking a random
    # action vs the action with the best Q
    if random.random() < explorationRate:
        # Random action
        action = env.action_space.sample()
    else:
        # Action with best Q for current state
        action = np.argmax(qTable[state])
    return action

def actionSelectFix(episode, state, explorationRate):
    if state == 4 or state == 13:
        return ACTION_KICK
    return ACTION_STRAIGHT

# Permutations of actions
def actionSelectPermute(episode, state, explorationRate):
    return np.argmax(qTable[state])

# This is a list of tables which represent the action to perform in a specific state
# So each table is like a Q-Table except that it contains the action number for the best action
# It is used to populate the Q-Table
permuteUsedBinStart = 2
permuteSecondDirectionBinEnd = xAccNumBins * 2 - permuteUsedBinStart - 1
permuteUsedBinCount = 10
permutationsTable = [perm for perm in itertools.product(range(numActions), repeat=permuteUsedBinCount)]
def permuteTableSetup(episode):
    for i, perm in enumerate(permutationsTable[episode % len(permutationsTable)]):
        if i < permuteUsedBinCount // 2:
            qTable[i+permuteUsedBinStart][1] = perm
        else:
            qTable[permuteSecondDirectionBinEnd-(i-permuteUsedBinCount // 2)][1] = perm

def permutesDone(episode):
    return episode > len(permutationsTable)

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
        
indHueMin = 0/360 
indHueMax = 100/360
kickIndicators = []
binBoundsAngles = [np.arcsin(np.clip(binBound / 9.81, -1, 1)) for binBound in xAccBinBounds]
binCentreAngles = [(binBoundsAngles[binBoundsIdx]+binBoundsAngles[binBoundsIdx-1])/2 for binBoundsIdx in range(1,len(binBoundsAngles))]

def doRender(numStreaks):
    if (not SHOW_RENDER) and (numStreaks != STREAK_LEN_WHEN_DONE-1):
        return
    from gym.envs.classic_control import rendering
    oldViewer = env.viewer
    env.render()
    if oldViewer is None:
        lineStart = -0.5
        lineEnd = -1
        textPosns = [-0.6, -0.9]
        for binBoundsAngle in binBoundsAngles:
            # Draw line
            x1 = np.sin(binBoundsAngle) * lineStart
            y1 = np.cos(binBoundsAngle) * lineStart
            x2 = np.sin(binBoundsAngle) * lineEnd
            y2 = np.cos(binBoundsAngle) * lineEnd
            segLine = rendering.make_polyline([(x1,y1),(x2,y2)])
            segLine.set_color(0/255, 0/255, 255/255)
            env.viewer.add_geom(segLine)
        for binCentreAngle in binCentreAngles:
            # Create kick indicator items
            kickDirnIndicators = []
            for i in range(numDirections):
                xT = np.sin(binCentreAngle) * textPosns[i]
                yT = np.cos(binCentreAngle) * textPosns[i]
                kickIndicator = rendering.make_circle(.05)
                kickTransform = rendering.Transform()
                kickTransform.set_translation(xT, yT)
                kickIndicator.add_attr(kickTransform)
                kickDirnIndicators.append(kickIndicator)
                env.viewer.add_geom(kickIndicator)
            kickIndicators.append(kickDirnIndicators)
    
    for i in range(len(binCentreAngles)):
        for j in range(numDirections):
            qIdx = i+1
            if j > 0:
                qIdx = 2*(len(binCentreAngles)+2)-i-2
            qRowRange = qTable[qIdx][0] - qTable[qIdx][1]
            if qRowRange <= -0.01:
                indHue = indHueMin
            elif qRowRange >= 0.01:
                indHue = indHueMax
            else:
                indHue = (indHueMax + indHueMin) / 2
            rgbColour = matplotlib.colors.hsv_to_rgb((indHue, 1, 1))
            kickIndicators[i][j].set_color(rgbColour[0], rgbColour[1], rgbColour[2])

    env.render()
    time.sleep(.01)

if __name__ == "__main__":
    learnToSwing()
