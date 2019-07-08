import sys, pygame, martypy

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

# Do this forever - until the user breaks out
while True:

    # Get the information about Marty's swing
    xAcc = myMarty.get_accelerometer('x')
    
    # Clear the screen then draw a shape to represent Marty
    screen.fill(black)
    martyX = screenWidth/2 + xAcc*screenWidth
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
