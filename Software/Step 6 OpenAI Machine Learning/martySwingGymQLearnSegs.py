# With inspiration from https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# -*- coding: utf-8 -*-

import gym
import gym_martyswing
import numpy as np
import time, math, random
import matplotlib.pyplot as plt
import matplotlib
import keyboard

# Create the MartySwing environment
env = gym.make('MartySwing-v0')

# Discrete actions
numActions = env.action_space.n # (kick, straight)
actionNames = ["Kick", "Straight", ""]
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

# Smoothing of values
obsList = []
obsSum = 0
obsWindowLen = 5

# Learning rate and exploration settings
EXPLORATION_RATE_MAX = 1
EXPLORATION_RATE_MIN = 0.01
EXPLORATION_RATE_DECAY_FACTOR = 3
LEARN_RATE_MAX = 0.5
LEARN_RATE_MIN = 0.05
LEARN_RATE_DECAY_FACTOR = 1
DISCOUNT_FACTOR = 0.9

# Goal and debug settings
EPISODE_MAX = 2000
TIME_MAX = 1000
STREAK_LEN_WHEN_DONE = 100
REWARD_SUM_GOAL = 110
LOG_DEBUG = True
LOG_DEBUG_FILE = "testruns/martySwingQLearnSegLog.txt"
SHOW_RENDER = True

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
        rewardSum = 0

        # Initial state
        statePrev = getObservationBinned(observation[0], xAccBinBounds)

        # Run the experiment over time steps
        t = 0
        while True:

            # Render the scene
            doRender(streaksNum)

            # Select an action
            action = actionSelect(statePrev, explorationRate)

            # Execute the action
            observation, reward, done, info = env.step(action)
            t += 1
            state = getObservationBinned(observation[0], xAccBinBounds)

            # Get the new state
            rewardInState = reward
            done = False
            while not done:
                # Get state
                state = getObservationBinned(observation[0], xAccBinBounds)
                if state != statePrev:
                    break

                # Render the scene
                doRender(streaksNum)

                # Repeat last action
                observation, reward, done, info = env.step(action)
                t += 1
                rewardInState += reward

            # Sum rewards
            rewardSum += rewardInState

            # Update the Q Table using the Bellman equation
            best_q = np.amax(qTable[state])
            qTable[statePrev, action] += learningRate*(rewardInState + DISCOUNT_FACTOR*(best_q) - qTable[statePrev, action])

            # Print data
            if logDebugFile is not None:
                # if debugActPrev != action:
                logDebugFile.write(f"{actionNames[action]} --- Ep {episode} t {t} statePrev {statePrev} state {state} rew {rewardInState} {'+' if rewardInState > 0 else '-'} bestQ {best_q} PE {info['PE']} KE {info['KE']} TE {info['PE']+info['KE']} theta {info['theta']} thetaMax {info['thetaMax']} v {info['v']} explRate {explorationRate} learnRate {learningRate} Streaks {streaksNum} \n")
                logDebugFile.write(dumpQTable(qTable))
                debugActPrev = action
                # ky = keyboard.read_key()
                # if ky == 'esc':
                #     exit(0)

            # Setting up for the next iteration
            statePrev = state

            if done or t > TIME_MAX:
                rewardTotal.append(info["thetaMax"])
                logStr = f"Episode {episode} finished after {t} rewardSum {rewardSum} thetaMax {info['thetaMax']} learnRate {learningRate} exploreRate {explorationRate} streakLen {streaksNum}"
                if logDebugFile:
                    logDebugFile.write("....." + logStr + "\n")
                if episode % 100 == 0:
                    print(dumpQTable(qTable))
                print(logStr)
                if (rewardSum >= REWARD_SUM_GOAL):
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

def actionSelectFix(state, explorationRate):
    if state == 4 or state == 13:
        return 0
    return 1

def getExplorationRate(t):
    # Exploration rate is a log function reducing over time
    return max(EXPLORATION_RATE_MIN, EXPLORATION_RATE_MAX * (1.0 - math.log10(t/EXPLORATION_RATE_DECAY_FACTOR+1)))

def getLearningRate(t):
    # Learning rate is a log function reducing over time
    return max(LEARN_RATE_MIN, LEARN_RATE_MAX * (1.0 - math.log10(t/LEARN_RATE_DECAY_FACTOR+1)))

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
    discreteVal = np.digitize(obsSum/len(obsList), bins)
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
    # print(dumpQTable(qTable))

    time.sleep(0.01)

if __name__ == "__main__":
    learnToSwing()
