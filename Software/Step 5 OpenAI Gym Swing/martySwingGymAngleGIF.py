import gym
import gym_martyswing
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Create the MartySwing
env = gym.make('MartySwing-v0')

# First action is to remain straight-legged
nextAction = 1

# Reset the Gym
env.dt = 0.1
env.thetaInit = np.radians(15)
env.l2 = 0.3
observation = env.reset()

# List of images for the GIF
framesAngle = []
tim = []
th = []
maxTime = 25

# Go through a number of swings
while(True):
    # Take the next action
    observation, reward, done, info = env.step(nextAction)

    # Render the image of marty swinging
    martyImage = Image.fromarray(env.render(mode='rgb_array'))

    # Add to data series
    tim.append(info["t"])
    th.append(np.degrees(info["theta"]))

    # Plot angle
    axes = plt.gca()
    axes.set_xlim([0,maxTime])
    axes.set_ylim([-60,60])
    plt.plot(tim, th, 'g')
    plt.suptitle("Marty Swing Simulator", fontsize=30)
    plt.ylabel('Swing Angle', fontsize=16)
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    plotImage = Image.frombytes('RGB', canvas.get_width_height(), 
                 canvas.tostring_rgb())
    wRatio = (martyImage.size[0]/float(plotImage.size[0]))
    hReq = int((float(plotImage.size[1])*float(wRatio)))
    plotImage = plotImage.resize((martyImage.size[0], hReq), Image.ANTIALIAS)

    # Plot together
    cropTop = 100
    cropBottom = 100
    totalWidth = martyImage.size[0]
    totalHeight = martyImage.size[1] + plotImage.size[1] - cropTop - cropBottom
    outImage = Image.new("RGB", (totalWidth, totalHeight))
    outImage.paste(martyImage, (0,-cropTop))
    outImage.paste(plotImage, (0,martyImage.size[1]-cropTop-cropBottom))

    # Add the combined image to a list
    framesAngle.append(outImage)

    # Select a new action based on the position in the swing cycle
    if (info["theta"] < np.radians(10) and info["theta"] > np.radians(-10)) and info["kickAngle"] < 0.01:
        # Kick if we are near the middle and haven't kicked already
        nextAction = 0
    elif (info["theta"] < np.radians(-14) or info["theta"] > np.radians(14)) and info["kickAngle"] > 0.01:
        # Straighten if we are near the top of our swing and not straight-legged already
        nextAction = 1

    # Check if we completed this test
    if info["t"] > maxTime and not done:
        fileName = 'MartySwingSim.gif'
        print("Saving GIF image")
        with open(fileName, 'wb') as outFile:
            im = Image.new('RGB', framesAngle[0].size)
            im.save(outFile, save_all=True, append_images=framesAngle)
        break

# Close the swing environment
env.close()
