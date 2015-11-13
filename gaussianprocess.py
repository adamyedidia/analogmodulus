import matplotlib.pyplot as p
from scipy.stats import multivariate_normal
import numpy as np

m = 4

covarianceMatrix = 0.001*np.array([[1 + (m+1)^2, m*(m+1)+1], [m*(m+1)+1, 1+m^2]])/(1+m^2)

integerCoefficientVec1 = np.array([1, -1])
integerCoefficientVec2 = np.array([-m, m+1])

integerMatrix = np.array([[1, -1], [-m, m+1]])

def inferMaxLikelihoodValues(xMod1, yMod1):
    global covarianceMatrix
    var = multivariate_normal(mean=[0,0], cov=covarianceMatrix)
    
    bestProb = 0
    bestValues = None
    
    for xOffset in range(-5, 6):
        for yOffset in range(-5, 6):
            candidate = var.pdf([xOffset+xMod1, yOffset+yMod1])
            if candidate > bestProb:
                bestValues = (xOffset+xMod1, yOffset+yMod1)
                bestProb = candidate
                
    return bestValues
        
def mod2OnLowInterval(x):
    return ((x+1)%2)-1

# takes as input a modded input on the interval [-1, 1]
def convertNumberToBits(x, numBits):
    
    if numBits == 0:
        return []
        
    elif x > 0:
        return [1] + convertNumberToBits(x*2-1, numBits-1)
    else:
        return [0] + convertNumberToBits(x*2+1, numBits-1)
        
def convertBitsToNumber(bits):
    if bits == []:
        return 0
    
    elif bits[0] == 1:
        return 0.5 + 0.5*convertBitsToNumber(bits[1:])
        
    else:
        return -0.5 + 0.5*convertBitsToNumber(bits[1:])           

def quantize(x, numBits):
    return convertBitsToNumber(convertNumberToBits(x, numBits))

#print convertBitsToNumber(convertNumberToBits(0.3, 2))
print quantize(0.23, 5)
print convertBitsToNumber([0,0,0])
print convertBitsToNumber([0,0,1])
print convertBitsToNumber([0,1,0])
print convertBitsToNumber([0,1,1])
print convertBitsToNumber([1,0,0])
print convertBitsToNumber([1,0,1])
print convertBitsToNumber([1,1,0])
print convertBitsToNumber([1,1,1])

        
n = 21

firstWaveform = []
secondWaveform = []

blackLine4 = [4]*n
blackLine2 = [2]*n
blackLine0 = [0]*n
blackLineMinus2 = [-2]*n
blackLineMinus4 = [-4]*n

#cov = np.array([[1, .5], [.5, 1]])

for i in range(n):
    newDataPoint = np.random.multivariate_normal([0, 0], covarianceMatrix)
        
    firstWaveform.append(newDataPoint[0])
    secondWaveform.append(newDataPoint[1])

firstWaveformReduced = [mod2OnLowInterval(x) for x in firstWaveform]
secondWaveformReduced = [mod2OnLowInterval(x) for x in secondWaveform]

firstWaveformQuantized = [quantize(x, 12) for x in firstWaveformReduced]
secondWaveformQuantized = [quantize(x, 12) for x in secondWaveformReduced]

#firstLinComb = [integerCoefficientVec1[0]*x + integerCoefficientVec1[1]*y for x, y in zip(firstWaveformQuantized, secondWaveformQuantized) 
#secondLinComb = [integerCoefficientVec2[0]*x + integerCoefficientVec2[1]*y for x, y in zip(firstWaveformQuantized, secondWaveformQuantized)]

#print np.dot(integerMatrix, np.array([[firstWaveformQuantized[0]], [secondWaveformQuantized[0]]]))

firstLinComb = [np.dot(integerMatrix, np.array([[x], [y]]))[0] for x,y in zip(firstWaveformQuantized, secondWaveformQuantized)]
secondLinComb = [np.dot(integerMatrix, np.array([[x], [y]]))[1] for x,y in zip(firstWaveformQuantized, secondWaveformQuantized)]

firstLinCombReduced = [mod2OnLowInterval(x) for x in firstLinComb]
secondLinCombReduced = [mod2OnLowInterval(x) for x in secondLinComb]

inverseIntegerMatrix = np.linalg.inv(integerMatrix)

estimateOfFirstWaveform = [np.dot(inverseIntegerMatrix, np.array([x, y]))[0] for x,y in zip(firstLinCombReduced, secondLinCombReduced)]
estimateOfSecondWaveform = [np.dot(inverseIntegerMatrix, np.array([x, y]))[1] for x,y in zip(firstLinCombReduced, secondLinCombReduced)]

p.clf()
p.plot(range(n), firstWaveform, "r-")
p.plot(range(n), secondWaveform, "b-")

p.savefig("originals.png")

p.clf()
p.plot(range(n), firstWaveformReduced, "r-")
p.plot(range(n), secondWaveformReduced, "b-")

p.savefig("originalsreduced.png")

p.clf()
p.plot(range(n), firstWaveformQuantized, "r-")
p.plot(range(n), secondWaveformQuantized, "b-")

p.savefig("originalquantized.png")

p.clf()
print firstLinComb, secondLinComb

p.plot(range(n), firstLinComb, "g-")
p.plot(range(n), secondLinComb, "m-")

p.savefig("lincombs.png")

p.clf()
p.plot(range(n), firstLinCombReduced, "g-")
p.plot(range(n), secondLinCombReduced, "m-")

p.savefig("lincombsreduced.png")

p.clf()

p.plot(range(n), estimateOfFirstWaveform, "r-")
p.plot(range(n), estimateOfSecondWaveform, "b-")
p.plot(range(n), firstWaveform, "r--")
p.plot(range(n), secondWaveform, "b--")

p.savefig("reconstructions.png")

#waveformSum = [x + y for x, y in zip(firstWaveform, secondWaveform)]
#waveformDiff = [x - y for x, y in zip(firstWaveform, secondWaveform)]    

#sumMod2 = [x % 2 for x in waveformSum]
#diffMod2 = [x % 2 for x in waveformDiff]

#firstWaveformMod1 = [((x+y)/2)%1 for x, y in zip(sumMod2, diffMod2)]
#secondWaveformMod1 = [((x-y)/2)%1 for x, y in zip(sumMod2, diffMod2)]
    
#inferredValues = [inferMaxLikelihoodValues(x,y) for x, y in zip(firstWaveformMod1, secondWaveformMod1)]

#inferredFirst = [x[0] for x in inferredValues]
#inferredSecond = [x[1] for x in inferredValues]    
    
#p.plot(range(n), firstWaveform, "r-")
#p.savefig("redonly.png")

#p.clf()
#p.plot(range(n), secondWaveform, "b-")    
#p.savefig("blueonly.png")

#p.plot(range(n), firstWaveform, "r-")    
#p.savefig("correlatedwaveforms.png")   
    
#p.clf()
#p.plot(range(n), waveformSum, "m-")
#p.savefig("sumonly.png")

#p.clf()
#p.plot(range(n), waveformDiff, "g-")
#p.savefig("diffonly.png")    

#p.clf()
#p.plot(range(n), waveformDiff, "g-")
#p.plot(range(n), waveformSum, "m-")
#p.savefig("sumanddiff.png")
    
#p.clf()
#p.plot(range(n), sumMod2, "m-")
#p.plot(range(n), diffMod2, "g-")    
#p.savefig("mods.png")    
    
#p.clf()
#p.plot(range(n), firstWaveformMod1, "r-")
#p.plot(range(n), secondWaveformMod1, "b-")
#p.savefig("originalmods.png")

    
#p.clf()
#p.plot(range(n), waveformDiff, "g-")    
#p.plot(range(n), waveformSum, "m-")
    
#p.plot(range(n), blackLine4, "k-")
#p.plot(range(n), blackLine2, "k-")
#p.plot(range(n), blackLine0, "k-")
#p.plot(range(n), blackLineMinus2, "k-")
#p.plot(range(n), blackLineMinus4, "k-")
#p.savefig("correlatedwaveformswithbins.png")

#p.clf()
#p.plot(range(n), firstWaveform, "r-")
#p.plot(range(n), secondWaveform, "b-")    
#p.plot(range(n), inferredFirst, "r--")
#p.plot(range(n), inferredSecond, correlatedwaveformswithguesses, "b--")
#p.savefig(".png")


