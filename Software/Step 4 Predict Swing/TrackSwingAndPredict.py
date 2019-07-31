import math

class MartySwingTracker:
    def __init__(self, estimatedPeriodSecs, estimatedStartAmplitude):
        # Estimated period
        self.estimatedPeriodSecs = estimatedPeriodSecs
        self.estimatedStartAmplitude = estimatedStartAmplitude
        # Window size must be an odd number
        self.numPrevValuesToKeep = 5
        # Time of first record seen
        self.firstRecTime = None
        # Period of swing
        self.swingPeriodSecs = self.estimatedPeriodSecs
        self.swingPeakLastSecs = 0
        self.swingAmplitude = estimatedStartAmplitude
        self.swingCentreBias = 0
        # Max adjustment factors
        self.adjustFactorPeriod = 0.01
        self.adjustFactorAmplitude = 0.5
        self.adjustFactorPeakSecs = 0.1
        self.adjustFactorBias = 0.1
        # Min and max values recorded
        self.minVal = 1
        self.maxVal = -1
        # Window of acclerometer and time values
        self.prevXAccValues = []
        self.prevTimeValues = []

    def isPeakOrNadir(self):
        # Only continue when there is enough data
        if len(self.prevXAccValues) < self.numPrevValuesToKeep:
            return (False, False, 0)
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
        return (isPeak, isNadir, (self.numPrevValuesToKeep-1)//2)

    def getExpected(self, timeSecs):
        predictedValue = math.sin(math.pi/2 + 2 * math.pi * (timeSecs - self.swingPeakLastSecs) / self.estimatedPeriodSecs) * self.swingAmplitude + self.swingCentreBias
        return predictedValue

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
        isPk, isNadir, pkIdx = self.isPeakOrNadir()
        if isNadir:
            # Record nadir value
            self.lastNadirValue = self.prevXAccValues[pkIdx]
        if isPk:
            # Get the time and value of this peak (it is not the most recent time as we have a window)
            thisPeakTime = self.prevTimeValues[pkIdx]
            thisPeakValue = self.prevXAccValues[pkIdx]
            
            # Amplitude adjustment
            self.swingAmplitude += (thisPeakValue - (self.swingAmplitude + self.swingCentreBias)) * self.adjustFactorAmplitude

            # Check for previous peak
            if self.swingPeakLastSecs == 0:
                # Set the peak time
                self.swingPeakLastSecs = thisPeakTime
            else:
                # Period adjustment
                self.swingPeriodSecs += (thisPeakTime - (self.swingPeakLastSecs + self.swingPeriodSecs)) * self.adjustFactorPeriod

                # Last peak adjustment
                self.swingPeakLastSecs += self.swingPeriodSecs
                self.swingPeakLastSecs += (thisPeakTime - self.swingPeakLastSecs) * self.adjustFactorPeakSecs 

                # Adjust bias level
                calcBiasLevel = (thisPeakValue + self.lastNadirValue) / 2
                self.swingCentreBias += (calcBiasLevel - self.swingCentreBias) * self.adjustFactorBias

        # Debug
        # print(timeSecs, xAcc, self.swingAmplitude, self.swingPeriodSecs, self.swingPeakLastSecs, self.swingCentreBias)
        return self.getExpected(timeSecs)


# Create the tracker
swingTracker = MartySwingTracker(1.2, 0.4)

# Read the data file
swingDataLines = []
with open("testruns/martySwingAndTime.txt", "r") as swingDataFile:
    swingDataLines = swingDataFile.readlines()

# Open output file for results
predictionOutFile = "testruns/martySwingAndPrediction.txt"
with open(predictionOutFile, "w+") as swingOutFile:

    # Write header line
    swingOutFile.write("t\tmeasuredXAcc\tpredictedXAcc\n")

    # Track the swing and predict future motion
    for swingDataLine in swingDataLines:
        swingDataFields = swingDataLine.split('\t')
        timeSecs = float(swingDataFields[0])
        xAcc = float(swingDataFields[1])
        predictedValue = swingTracker.newData(timeSecs, xAcc)
        swingOutFile.write(f"{timeSecs:.3f}\t{xAcc:.4f}\t{predictedValue:.4f}\n")

    # Done
    print(f"Predicted results written to {predictionOutFile}")
    print(f"Open the file in a spreadsheet like Google Sheets and plot columns B and C on a line chart")
