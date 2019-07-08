import sys, martypy
from datetime import datetime

# Initialise Marty at his IP address
myMarty = martypy.Marty('socket://192.168.86.41') # Change IP address to your one

# Create file for the data
with open("martySwingAndTime.txt", "w+") as swingData:

    # Track last value
    lastVal = 0

    # Collect data for a fixed time
    timeStart = datetime.now()
    while (datetime.now() - timeStart).total_seconds() < 60:

        # Get the information about Marty's swing
        xAcc = myMarty.get_accelerometer('x')
        
        # Record the data
        if lastVal != xAcc:
            elapsedSecs = (datetime.now() - timeStart).total_seconds()
            swingData.write(f"{elapsedSecs:.3f}\t{xAcc:.4f}\n")
            lastVal = xAcc

