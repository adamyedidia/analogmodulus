import numpy as np
import matplotlib.pyplot as p
from math import pi, sin, sqrt

def sinc(x):
    if x == 0:
        return 1
    else:
        return sin(x) / x
        
def average(x):
    return float(sum(x))/len(x)        
        
def variance(x):
    averageX = average(x)
    
    return average([abs((i - averageX)*(i - averageX)) for i in x])        
        
def dot(x, y):
    return sum([i*j for i,j in zip(x,y)])
        
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

def mod2OnLowInterval(x):
    return ((x+1)%2)-1
    
def mod1OnLowInterval(x):
    return ((x+0.5)%1)-0.5
    
def modHalfOnLowInterval(x):
    return ((x+0.25)%0.5)-0.25
        
P = 100

L=32

n=2**7

D = 0.00001

numSamples = n*L

waveform = []

for i in range(-int(numSamples/2), int(numSamples/2)+1):
    waveform.append(np.random.normal(0, 1))
    
transformedWaveform = np.fft.fft([x for x in waveform])    
    
#indexedTransform = []
        
#counter = 0        
#for i in range(-int(numSamples/2), int(numSamples/2)+1):
#    index = float(i)/n
#    indexedTransform.append((index, transformedWaveform[counter]))
#    counter += 1

prunedTransform = [val*((float(i) < n/2.) or (float(i) > numSamples - n/2.)) \
    for i, val in enumerate(transformedWaveform)]
        
#p.plot(range(-int(numSamples/2), int(numSamples/2)), prunedTransform, "b-")


#p.plot(range(-int(numSamples/2), int(numSamples/2)+1), prunedTransform)

predictions = []

RSY = np.array([sinc(float(i)/L) for i in range(P)])

RYY = np.array([[sinc(float(i)/L) for i in range(j, 0, -1)] + \
                [D*L+1] + \
                [sinc(float(i)/L) for i in range(1, P-j, 1)] for j in range(P)])

predictionVec = np.dot(np.linalg.inv(RYY) , RSY)

returnedWaveform = np.fft.ifft(prunedTransform)  
#scaledUpWaveform = [x*1 for x in returnedWaveform]
scaledUpWaveform = [x*sqrt(L) for x in returnedWaveform]



var = variance(scaledUpWaveform)
print "var", var
#print variance([x[1] for x in waveform])

moddedWaveform = [mod2OnLowInterval(np.real(x)) for x in scaledUpWaveform]

#moddedWaveform = scaledUpWaveform

quantizedWaveform = [quantize(x, 4) for x in moddedWaveform]

#quantizedWaveform = moddedWaveform

p.plot(range(numSamples+1), quantizedWaveform, "b-")
p.show()

for i in range(P, numSamples):
    pPreviousSamples = returnedWaveform[i-P:i]
#    print pPreviousSamples
#    print RSY, pPreviousSamples
    predictions.append(dot(predictionVec, pPreviousSamples))  
    
p.plot(range(numSamples+1), returnedWaveform, "b-")
p.savefig("oversampledwaveform.png")
    
p.clf()    
    
#p.plot(range(-int(numSamples/2), int(numSamples/2)+1)[100:130], returnedWaveform[100:130], "b-")
p.plot(range(numSamples+1), returnedWaveform, "b-")
p.plot(range(numSamples-100), predictions, "r-")

p.savefig("oversampledwaveformwithprediction.png")

p.clf()
p.plot(range(numSamples-100), [(x-y)**2 for x,y in zip(returnedWaveform[:-101], predictions)])
p.savefig("squarederror.png")
