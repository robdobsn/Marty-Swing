import sys, martypy
from datetime import datetime

# Initialise Marty at his IP address
myMarty = martypy.Marty('socket://192.168.86.41') # Change IP address to your one

# Create file for the data
with open("martySwing.txt", "w+") as swingData:

    # Collect data for a fixed time
    timeStart = datetime.now()
    while (datetime.now() - timeStart).total_seconds() < 20:

        # Get the information about Marty's swing
        xAcc = myMarty.get_accelerometer('x')
        
        # Record the data
        swingData.write(f"{xAcc:.4f}\n")

