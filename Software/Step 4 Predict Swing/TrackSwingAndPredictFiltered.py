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
        self.adjustFactorPeriod = 0.1
        self.adjustFactorAmplitude = 0.5
        self.adjustFactorPeakSecs = 0.1
        self.adjustFactorBias = 0.1
        # Min and max values recorded
        self.minVal = 1
        self.maxVal = -1
        # Window of acclerometer and time values
        self.prevUnfilteredValues = []
        self.prevFilteredValues = []
        # Setup filter
        self.filterSetup()

    def filterSetup(self):
        # The following coefficients generated with https://www.earlevel.com/main/2013/10/13/biquad-calculator-v2/
        # # Using 10Hz sample rate, 1.2 Hz cutoff frequency, 0.7 Q and 6db Gain
        # self.a0 = 0.091315
        # self.a1 = 0.182629
        # self.a2 = 0.091315
        # self.b1 = -0.982403
        # self.b2 = 0.347661
        # # Using 10Hz sample rate, 1.0 Hz cutoff frequency, 0.7 Q and 6db Gain
        # self.a0 = 0.06745508395870334
        # self.a1 = 0.13491016791740668
        # self.a2 = 0.06745508395870334
        # self.b1 = -1.1429772843080923
        # self.b2 = 0.41279762014290533
        # # Using 10Hz sample rate, 0.85 Hz centre frequency, 10.0 Q
        # self.a0 = 0.067900771261357
        # self.a1 = 0.135801542522714
        # self.a2 = 0.067900771261357
        # self.b1 = -1.6787562315670543
        # self.b2 = 0.9503593166124826
        # Using 10Hz sample rate, 0.85 Hz centre frequency, 3.0 Q
        self.a0 = 0.06418363201334885
        self.a1 = 0.1283672640266977
        self.a2 = 0.06418363201334885
        self.b1 = -1.5868549090890356
        self.b2 = 0.8435894371424311

    def filterSample(self, inWin, outWin):
        t = len(inWin)-1
        if t >= 2:
            return self.a0*inWin[t][1] + self.a1*inWin[t-1][1] + self.a2*inWin[t-2][1] - self.b1*outWin[t-1][1] - self.b2*outWin[t-2][1]
        return inWin[t][1]

    def isPeakOrNadir(self, testValues):
        # Only continue when there is enough data
        if len(testValues) < self.numPrevValuesToKeep:
            return (False, False, 0)
        isPeak = True
        isNadir = True
        # Check for rising / falling up to central point
        for i in range(1,(self.numPrevValuesToKeep+1)//2):
            if testValues[i][1] < testValues[i-1][1]:
                isPeak = False
            elif testValues[i][1] > testValues[i-1][1]:
                isNadir = False
        # Check for falling / rising from the central point
        for i in range((self.numPrevValuesToKeep+1)//2,self.numPrevValuesToKeep):
            if testValues[i][1] > testValues[i-1][1]:
                isPeak = False
            elif testValues[i][1] < testValues[i-1][1]:
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
        self.prevUnfilteredValues = self.prevUnfilteredValues[-(self.numPrevValuesToKeep-1):]
        self.prevUnfilteredValues.append((timeSecs, xAcc))
        self.prevFilteredValues = self.prevFilteredValues[-(self.numPrevValuesToKeep-1):]
        filteredSample = self.filterSample(self.prevUnfilteredValues, self.prevFilteredValues)
        self.prevFilteredValues.append((timeSecs, filteredSample))
        # Find peaks
        isPk, isNadir, pkIdx = self.isPeakOrNadir(self.prevFilteredValues)
        if isNadir:
            # Record nadir value
            self.lastNadirValue = self.prevFilteredValues[pkIdx][1]
        if isPk:
            # Get the time and value of this peak (it is not the most recent time as we have a window)
            thisPeakTime = self.prevFilteredValues[pkIdx][0]
            thisPeakValue = self.prevFilteredValues[pkIdx][1]
            
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
        # expectedValue = self.getExpected(timeSecs)
        # print(timeSecs, xAcc, self.swingAmplitude, self.swingPeriodSecs, self.swingPeakLastSecs, self.swingCentreBias, expectedValue, self.prevFilteredValues[len(self.prevFilteredValues)-1][1])
        return (self.getExpected(timeSecs), self.prevFilteredValues[len(self.prevFilteredValues)-1][1],
                              0 if len(self.prevUnfilteredValues) < 4 else self.prevUnfilteredValues[len(self.prevUnfilteredValues)-4][1])


# Create the tracker
swingTracker = MartySwingTracker(1.2, 0.4)

# Read the data file
swingDataLines = []
with open("testruns/martySwingAndTime.txt", "r") as swingDataFile:
    swingDataLines = swingDataFile.readlines()

# Open output file for results
predictionOutFile = "testruns/martySwingAndPredictionFiltered.txt"
with open(predictionOutFile, "w+") as swingOutFile:

    # Write header line
    swingOutFile.write("t\tmeasuredXAcc\tpredictedXAcc\n")

    # Track the swing and predict future motion
    for swingDataLine in swingDataLines:
        swingDataFields = swingDataLine.split('\t')
        timeSecs = float(swingDataFields[0])
        xAcc = float(swingDataFields[1])
        predictedValue, filteredValue, delayedValue = swingTracker.newData(timeSecs, xAcc)
        swingOutFile.write(f"{timeSecs:.3f}\t{xAcc:.4f}\t{filteredValue:.4f}\n")

    # Done
    print(f"Predicted results written to {predictionOutFile}")
    print(f"Open the file in a spreadsheet like Google Sheets and plot columns B and C on a line chart")
