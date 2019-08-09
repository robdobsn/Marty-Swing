#Submission of Final Project Report : Anirudh Topiwala
# Implementing Q learning On cart Pole Problem. 
# This code is made by taking reference of https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# -*- coding: utf-8 -*-
import gym
import gym_martyswing
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import keyboard

## Initialize the "Cart-Pole" environment
env = gym.make('martyswing-v0')

## Defining the environment related constants

# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (nothing, kick, unkick)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Discrete bounds for observation
XACC_BUCKET_BOUNDS = [STATE_BOUNDS[0][0]*0.5, STATE_BOUNDS[0][0]*0.1, STATE_BOUNDS[0][1]*0.1, STATE_BOUNDS[0][1]*0.5]
# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (len(XACC_BUCKET_BOUNDS)+1,)  # (xAcc)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

## Defining the simulation related constants
NUM_EPISODES = 1200
MAX_T = 1000
STREAK_TO_END = 100
SOLVED_T = 5
DEBUG_MODE = False

def simulate():

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0
    rewardTotal = []

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()
        rewardSum = 0

        # the initial state
        # state_0 = state_to_bucket(obv)
        state_0 = xAccToBucket(obv[0])

        for t in range(MAX_T):
            if num_streaks > STREAK_TO_END - 2:
                env.render()
                time.sleep(0.1)

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)
            rewardSum += reward
            state = xAccToBucket(obv[0])

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print(f"Ep {episode} t {t} Act {action} xAccBin {state[0]} rew {reward} bestQ {best_q} explRate {explore_rate} learnRate {learning_rate} Streaks {num_streaks}")
                # print("\nEpisode = %d" % episode)
                # print("t = %d" % t)
                # print("Action: %d" % action)
                # print("State: %s" % str(state))
                # print("Reward: %f" % reward)
                # print("Best Q: %f" % best_q)
                # print("Explore rate: %f" % explore_rate)
                # print("Learning rate: %f" % learning_rate)
                # print("Streaks: %d" % num_streaks)
                print(q_table)
                ky = keyboard.read_key()
                if ky == 'esc':
                    exit(0)

            if done:
                rewardTotal.append(rewardSum)
                print("Episode %d finished after %f time steps rewardSum %f learnRate %f exploreRate %f" % (episode, t, rewardSum, learning_rate, explore_rate))
                if DEBUG_MODE:
                    print(q_table)
                if (rewardSum >= SOLVED_T):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

        # It's considered done when it's solved over 100 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

    print(q_table)
    plt.plot(rewardTotal, 'p')
    plt.show()


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t+1)/25.0)))    #using Logrithmic decaying explore rate

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25.0)))  #using Logrithmic decaying learning rate

def xAccToBucket(xAcc):
    for i in range(len(XACC_BUCKET_BOUNDS)):
        if xAcc < XACC_BUCKET_BOUNDS[i]:
            return (i,)
    return (len(XACC_BUCKET_BOUNDS),)

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    simulate()
