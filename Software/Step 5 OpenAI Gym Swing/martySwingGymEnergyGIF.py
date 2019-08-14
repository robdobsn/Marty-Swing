import gym
import gym_martyswing
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Change this to set if Marty pumps his swing
martyPumps = True

# Create the MartySwing
env = gym.make('MartySwing-v0')

# First action is to remain straight-legged
nextAction = 1

# Reset the Gym
env.dt = 0.03
env.thetaInit = np.radians(40)
observation = env.reset()

# List of images for the GIF
framesEnergy = []
tim = []
ke = []
pe = []
maxTime = 1.4
if martyPumps:
    maxTime = 2.8

# Go through a number of swings
while(True):
    # Take the next action
    observation, reward, done, info = env.step(nextAction)

    # Render the image of marty swinging
    martyImage = Image.fromarray(env.render(mode='rgb_array'))

    # Add to data series
    tim.append(info["t"])
    pe.append(info["PE"])
    ke.append(info["KE"]) 

    # Plot energy
    plt.clf()
    axes = plt.gca()
    axes.set_xlim([0, maxTime])
    axes.set_ylim([0, 0.37 if not martyPumps else 0.45])
    plt.plot(tim, ke, 'b', label="Kinetic Energy")
    plt.plot(tim, pe, 'g', label="Potential Energy")
    plt.legend(loc='upper left')
    if martyPumps:
        plt.suptitle("Energy Added Pumping a Swing", fontsize=20)
    else:
        plt.suptitle("Energy Conversion in a Swing", fontsize=20)
    plt.ylabel('Energy', fontsize=16)
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
    framesEnergy.append(outImage)

    # Select a new action based on the position in the swing cycle
    if (info["theta"] < np.radians(10) and info["theta"] > np.radians(-10)) and info["kickAngle"] < 0.01:
        # Kick if we are near the middle and haven't kicked already
        nextAction = 0
    elif (info["theta"] < np.radians(-35) or info["theta"] > np.radians(35)) and info["kickAngle"] > 0.01:
        # Straighten if we are near the top of our swing and not straight-legged already
        nextAction = 1

    # Check if we completed this test
    if info["t"] > maxTime and not done:
        print("Saving GIF image")
        with open('MartySwingEnergy.gif' if not martyPumps else 'MartySwingPumpEnergy.gif', 'wb') as outFile:
            im = Image.new('RGB', framesEnergy[0].size)
            im.save(outFile, save_all=True, append_images=framesEnergy)
        break

# Close the swing environment
env.close()
