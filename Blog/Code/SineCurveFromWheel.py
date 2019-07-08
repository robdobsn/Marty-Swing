GlowScript 2.7 VPython

# Wheel radius
r = 0.3

# Angular velocity
omega = 1

# Keep track of elapsed time
t = 0

# Time interval for the simulation
dt = 0.01
animDt = dt

# Reversal point
tReverse = 2 * pi

# Start vector for wheel
startVec = vector(-0.9,0,0)

# Wheel
wheel = ring(pos=startVec, axis=vector(0,0,1),
      radius=r, thickness=0.02)
      
# Point on the wheel
wheelPoint = sphere(pos=startVec, radius=.05, color=color.red)

# Wheel position vector
wheelVec = startVec

# Curve
cStartVec = startVec + vector(-r,0,0)
c = curve(pos = [cStartVec, cStartVec], color=color.blue)
cxScale = 0.3

# Track line
trackLineExtension = vector(0.1,0,0)
trackLine = curve(pos = [cStartVec-trackLineExtension, cStartVec+trackLineExtension], color=color.green)

# Center initially
scene.center = vector(0,0,0)
scene.range = 1
scene.background = vector(240/255,241/255,242/255)

# Animate    
while True:

  # Animation update rate
  rate(1 / animDt)
  
  # Update theta
  theta = omega * t

  # Update wheel position
  wheelVec = startVec + vector(theta * r, 0, 0)
  
  # Update the visual representation of the wheel
  wheel.pos = wheelVec
  wheelPoint.pos = wheelVec + r * vector(-cos(theta),sin(theta),0)
  
  # Update the curve
  curCurvePoint = cStartVec + vector(theta*cxScale, r*sin(theta), 0)
  if dt > 0:
    c.append(curCurvePoint)
  else:
    c.pop()
    
  # Update the trackline
  trackLine.pop()
  trackLine.pop()
  trackLine.append(curCurvePoint - trackLineExtension)
  trackLine.append(wheelPoint.pos + trackLineExtension)

  # Increment simulation time
  t=t+dt
  
  # Reverse direction
  if t >= tReverse:
    t = tReverse
    dt = -dt
  else if t <= 0:
    t = 0
    dt = -dt
    c.clear()
    c.append(cStartVec, cStartVec)
