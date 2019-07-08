GlowScript 2.7 VPython

# Starting length of the string
L = 1

# Gravity factor for Earth = 9.81, Moon = 1.65
# Mars = 3.8, Jupiter = 24.5
g = 9.81

# Keep track of elapsed time
t = 0

# Time interval for the simulation
dt = 0.01

# Starting angle 
thetaStart = 50*pi/180

# String attachment point
top=sphere(pos=vector(0,0,0), radius=.02)

# Initial ball and string position
ball=sphere(pos=L*vector(sin(thetaStart), -cos(thetaStart),0),radius=.05, color=color.red)
string=cylinder(pos=top.pos, axis=(ball.pos-top.pos), radius=0.01, color=color.yellow)
attach_trail(ball, retain = 20, color=color.black)

# Track the top point of the swing
thetaDelta = 0
topCount = 0
swingStart = None
swingPeriod = 0
lengthChangeCount = 0

# Buttons to make the string longer and shorter
button(text="Longer String", pos=scene.title_anchor, bind=StringLonger)
button(text="Shorter String", pos=scene.title_anchor, bind=StringShorter)

# Center initially
scene.center = vector(0.3,-0.7,0)
scene.range = 1
scene.background = vector(240/255,241/255,242/255)

# String length label
labelColor = vec(0,0.5,0)
lengthLabel = label(pos=top.pos+string.axis/4, text="Length 1 m", xoffset=10, 
            yoffset=12, xoffset=20, height=14, color=labelColor)

# Function called on button press to make string longer
def StringLonger():
  global L, lengthChangeCount, swingStart
  if L <= 1.5:
    L = L + 0.1
    lengthChangeCount = 2
    periodLabel.text = "Period ? s"

# Function called on button press to make string shorter
def StringShorter():
  global L, lengthChangeCount, swingStart
  if L >= 0.3:
    L = L - 0.1
    lengthChangeCount = 2
    periodLabel.text = "Period ? s"
    swingStart = None

# Animate    
while True:

  # Animation update rate
  rate(1/dt)
  
  # Angular velocity
  omega = sqrt(g / L)

  # Update theta
  prevTheta = theta
  theta=thetaStart * sin(omega * t)
  prevThetaDelta = thetaDelta
  thetaDelta = theta-prevTheta

  # Update the visual representation of the ball
  ball.pos=L*vector(sin(theta),-cos(theta),0)
  
  # Update the string
  string.axis=ball.pos-top.pos
  
  # Update the string label
  lengthLabel.pos = top.pos+string.axis/4
  lengthLabel.text = "Length {:.1f} m".format(L)

  # Check for top point
  if thetaDelta < 0 and prevThetaDelta > 0 and lengthChangeCount == 0:

    # First swing so add label
    if topCount == 0:
      timeLabel = label(pos=ball.pos, text='0', xoffset=10, 
            yoffset=10, height=14, color=labelColor)
            
    # Show the time since the start of the swing
    timeLabel.pos = ball.pos

    # Show the period of the swing if we've completed at least one swing
    if topCount > 0:

      # Check if we have completed at least one swing
      if topCount == 1:

        # Make a label for the period of the swing
        periodLabel = label(pos=ball.pos, text="", xoffset=10, 
            yoffset=-10, height=14, color=labelColor)

      # Calculate the swing period
      if not (swingStart is None):
        swingPeriod = t - swingStart
        periodLabel.pos = ball.pos
        periodLabel.text = "Period " + "{:.1f}".format(swingPeriod) + " s"

    # Swing start
    swingStart = t
    
    # Count the swings
    topCount += 1

  # No longer check for length change
  if lengthChangeCount > 0:
    lengthChangeCount -= 1
  
  # Update the time label to show the elapsed time in the swing
  if not (swingStart is None) and topCount > 0:
    timeLabel.text = "{:.1f}".format(t - swingStart) + " s"

  # Increment simulation time
  t=t+dt
