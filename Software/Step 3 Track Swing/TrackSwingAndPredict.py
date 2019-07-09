import math

class MartySwingTracker:
    def __init__(self, estimatePeriodSecs):
        # Estimated period
        self.estimatePeriodSecs = estimatePeriodSecs
        # Window size must be an odd number
        self.numPrevValuesToKeep = 5
        # Time of first record seen
        self.firstRecTime = None
        # Period of swing
        self.swingPeriodSecs = estimatePeriodSecs
        self.swingPhaseOffsetSecs = 0
        self.swingAmplitude = None
        # Min and max values recorded
        self.minVal = 1
        self.maxVal = -1
        # Window of acclerometer and time values
        self.prevXAccValues = []
        self.prevTimeValues = []
        # 
        # self.xAccOffset = 0
        # self.cycleStartSecs = 0
        # self.findingStartCycle = False
        # self.lastPeakTime = None
        # self.lastPeakNegative = False
        # self.peakTimesSum = 0
        # self.peakTimesCount = 0
        # self.periodAvg = 1.0

    def isPeakOrNadir(self):
        # Only continue when there is enough data
        if len(self.prevXAccValues) < self.numPrevValuesToKeep:
            return False
        isPeak = True
        isNadir = True
        # Check for rising / falling up to central point
        for i in range(1,(self.numPrevValuesToKeep+1)//2):
            if self.prevXAccValues[i] < self.prevXAccValues[i-1]:
                isPeak = False
            elif self.prevXAccValues[i] > self.prevXAccValues[i-1]:
                isNadir = False
        # Check for falling / rising from the central point
        for i in range((self.numPrevValuesToKeep+1)//2,self.numPrevValuesToKeep):
            if self.prevXAccValues[i] > self.prevXAccValues[i-1]:
                isPeak = False
            elif self.prevXAccValues[i] < self.prevXAccValues[i-1]:
                isNadir = False
        return (isPeak or isNadir, (self.numPrevValuesToKeep-1)//2)

    def newData(self, timeSecs, xAcc):
        # Record the time of the first record
        if self.firstRecTime is None:
            self.firstRecTime = timeSecs
        # Add the new values to the window
        self.prevXAccValues = self.prevXAccValues[-(self.numPrevValuesToKeep-1):]
        self.prevXAccValues.append(xAcc)
        self.prevTimeValues = self.prevTimeValues[-(self.numPrevValuesToKeep-1):]
        self.prevTimeValues.append(timeSecs)
        # Find peaks
        isPk, pkIdx = self.isPeakOrNadir():
        if isPk:
            # Get the time of this peak (it is not the most recent time as we have a window)
            thisPeakTime = self.prevTimeValues[pkIdx]


            
            # Count up the peaks
            if self.lastPeakTime is not None:
                self.peakTimesSum += thisPeakTime - self.lastPeakTime
                self.peakTimesCount += 1
            self.lastPeakTime = thisPeakTime
            self.lastPeakNegative = xAcc < 0
        if timeSecs - self.firstRecTime > self.secsForPeak:
            # Find the offset (or bias) of the accelerometer readings
            self.xAccOffset = (self.maxVal + self.minVal) / 2
            self.initialSynch = False
            if self.peakTimesCount > 0:
                self.periodAvg = 2 * self.peakTimesSum / self.peakTimesCount
        if self.initialSynch:
            pass
        # Track the max/min values
        self.minVal = min(self.minVal, xAcc)
        self.maxVal = max(self.maxVal, xAcc)
        # # Find a zero-crossing point in the positive direction
        # elif self.findingStartCycle:
        #     if self.lastXAccValue < self.xAccOffset and xAcc > self.xAccOffset:
        #         self.cycleStartSecs = timeSecs
        #         self.findingStartCycle = False
        #     self.lastXAccValue = xAcc
        print(timeSecs, xAcc, self.minVal, self.maxVal, self.xAccOffset, self.cycleStartSecs, self.peakTimesSum, self.peakTimesCount, self.periodAvg)
        if self.initialSynch:
            return xAcc
        amplitude = (-1 if self.lastPeakNegative else 1) * (self.maxVal - self.minVal) / 2
        predictedValue = math.cos(2 * math.pi * (timeSecs - self.lastPeakTime) / self.periodAvg) * amplitude + self.xAccOffset
        return predictedValue

lastSwingVal = 0
swingDataLines = []

swingTracker = MartySwingTracker(1.2)

with open("martySwingAndTime.txt", "r") as swingDataFile:
    swingDataLines = swingDataFile.readlines()

with open("martySwingAndPrediction.txt", "w+") as swingOutFile:
    for swingDataLine in swingDataLines:
        swingDataFields = swingDataLine.split('\t')
        timeSecs = float(swingDataFields[0])
        xAcc = float(swingDataFields[1])
        predictedValue = swingTracker.newData(timeSecs, xAcc)
        swingOutFile.write(f"{timeSecs:.3f}\t{xAcc:.4f}\t{predictedValue:.4f}\n")
