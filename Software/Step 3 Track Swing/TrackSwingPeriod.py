class MartySwingPeriodTracker:
    def __init__(self, estimatePeriodSecs):
        # Variables
        self.estimatePeriodSecs = estimatePeriodSecs
        self.lastXAccValue = 1
        self.crossingTimeSecsCount = 0
        self.crossingTimeSecsLast = 0
        self.crossingTimeSecsFirst = None

    def newData(self, timeSecs, xAcc):

        # Find a zero-crossing point in the positive direction
        if self.lastXAccValue <= 0 and xAcc > 0:
            if self.crossingTimeSecsFirst is not None:
                # Try to avoid "wobble" by ensuring we don't count two crossings very close together
                timeSinceLastCrossing = timeSecs - self.crossingTimeSecsLast
                if timeSinceLastCrossing > self.estimatePeriodSecs / 2:
                    self.crossingTimeSecsLast = timeSecs
                    self.crossingTimeSecsCount += 1
            else:
                # Record the first time we go from negative to positive acceleration
                self.crossingTimeSecsFirst = timeSecs
                self.crossingTimeSecsLast = timeSecs

        # Remember the last xAcc value
        self.lastXAccValue = xAcc

    def getPeriod(self):
        if self.crossingTimeSecsCount > 0:
            return (self.crossingTimeSecsLast - self.crossingTimeSecsFirst) / self.crossingTimeSecsCount
        return 0

# Create the tracker
swingTracker = MartySwingPeriodTracker(1.2)

# Read the data file
swingDataLines = []
with open("testruns/martySwingAndTime.txt", "r") as swingDataFile:
    swingDataLines = swingDataFile.readlines()

# Track the swing period
lastSwingVal = 0
for swingDataLine in swingDataLines:
    # Extract the time and accelerometer values from the text on the line
    swingDataFields = swingDataLine.split('\t')
    timeSecs = float(swingDataFields[0])
    xAcc = float(swingDataFields[1])
    swingTracker.newData(timeSecs, xAcc)

# Print the period we've found
print(f"Period of Marty's swing is {swingTracker.getPeriod():.02f} seconds")
