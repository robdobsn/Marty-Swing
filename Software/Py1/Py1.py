import martypy
import time

mymarty = martypy.Marty('socket://192.168.86.43') # Change IP accordingly
mymarty.hello()  # Move to zero positions and wink

for i in range(1000):
    time.sleep(.01)
    x = mymarty.get_accelerometer('x')
    print(x)
