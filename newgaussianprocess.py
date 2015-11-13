import numpy as np
import matplotlib.pyplot as p

# includes x but does not include y
def interpolate(x, y, numPoints):
    step = float(y-x)/numPoints

    returnList = []

    for i in range(numPoints):
        returnList.append(x + i*step)
        
    return returnList

def getNextValue(prevValue, var):
    return np.random.normal(prevValue, var)
    
numCorrectPoints = 4    
numInterpolatedPoints = 7
n = (numCorrectPoints-1)*numInterpolatedPoints
correctPoints = []

for i in range(numCorrectPoints):
    correctPoints.append(np.random.normal(0, 1))

waveform = []
    
currentPointIndex = 0    
while currentPointIndex < len(correctPoints)-1:
    waveform += interpolate(correctPoints[currentPointIndex], correctPoints[currentPointIndex+1], numInterpolatedPoints)
    currentPointIndex += 1    
    
print waveform    
p.plot(range(n), waveform)

p.savefig("waveform.png")