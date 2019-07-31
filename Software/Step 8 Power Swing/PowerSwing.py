import pygame, martypy, time

# Initialise PyGame
pygame.init()

# Initialise Marty at his IP address
myMarty = martypy.Marty('socket://192.168.86.41') # Change IP address to your one

# Screen to display swing on
screenSize = screenWidth, screenHeight = 800, 600
screen = pygame.display.set_mode(screenSize)

# Size of shape to use to represent Marty
martySizeX = martySizeY = screenHeight//2.5

# Colours
black = (0,0,0)
martyColour = (37,167,253)

# Remember the last x accelerometer value
lastAccX = None
lastAccXAvg = 0
lastAccXDiff = None
lastKickTime = time.time()
smoothWindowLength = 5
lastAccelerometerValues = []

# Do this forever - until the user breaks out
while True:

    # Get the information about Marty's swing
    accX = myMarty.get_accelerometer('x')

    # Check for a change of direction
    if lastAccX is not None and lastAccX != accX:

        # Smooth out accelerometer values
        lastAccelerometerValues = lastAccelerometerValues[-smoothWindowLength:]
        lastAccelerometerValues.append(accX)
        if len(lastAccelerometerValues) > smoothWindowLength:

            # Get the average of the last N values
            accXSum = 0
            for prevAcc in lastAccelerometerValues:
                accXSum += prevAcc
            accXAvg = accXSum / smoothWindowLength

            # Check for difference
            newAccXDiff = accXAvg - lastAccXAvg

            # Simply kick if we were going up and aren't anymore
            if lastAccXDiff is not None:
                timeNow = time.time()
                timeSinceLastKickSecs = timeNow - lastKickTime
                print(f"AccXAvg {accXAvg} LastAccXAvg {lastAccXAvg} Diff {newAccXDiff} LastDiff {lastAccXDiff}")
                if newAccXDiff > 0 and lastAccXDiff < 0 and timeSinceLastKickSecs > 0.3:
                    print(f"Kick forwards {time.time()-lastKickTime}")
                    myMarty.move_joint(0,10,10)
                    myMarty.move_joint(3,10,10)
                    lastKickTime = time.time()
                elif newAccXDiff < 0 and lastAccXDiff > 0 and timeSinceLastKickSecs > 0.3:
                    print(f"Kick back {time.time()-lastKickTime}")
                    myMarty.move_joint(0,-10,10)
                    myMarty.move_joint(3,-10,10)
                    lastKickTime = time.time()
            # Save the difference for next time
            lastAccXDiff = newAccXDiff
            lastAccXAvg = accXAvg

    # Save the value for next time round
    lastAccX = accX
    
    # Clear the screen then draw a shape to represent Marty
    screen.fill(black)
    martyX = screenWidth/2 + accX*screenWidth
    pygame.draw.rect(screen, martyColour, (martyX-martySizeX/2, screenHeight//2, martySizeX, martySizeY), 10)

    # Flip to the screen to create animation
    pygame.display.flip()

    # Check if the user has asked to quit
    quitRequired = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            quitRequired = True
    if quitRequired:
        break

# Now that we've exited the loop let's quit
pygame.quit()
