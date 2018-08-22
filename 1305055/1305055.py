import numpy as np
import ast
# import scipy
import math

# read

file = open('config.txt', 'r')
file2 = open('train.txt', 'r')
file3 = open('test.txt', 'r')



n = int(file.readline()[0])
weights = file.readline()
variance = float(file.readline())
w_read = np.array(weights.split(" "))
w = np.zeros(len(w_read))

for i in range(len(w)):
    w[i] = ast.literal_eval(w_read[i])
#print(w)

trainSequence = file2.readline()
#print(trainSequence)

testSequence = file3.readline()
#print(testSequence)

numberOfStates = 2**int(n)
priorProbabilities = np.zeros((numberOfStates, 1))
#print(priorProbabilities)


# Prior probability calculation
for i in range(len(trainSequence) - n + 1):
    subSequence = trainSequence[i:i+n]
    #print(i, subSequence)

    idx = int(subSequence, 2)
    #print(idx)

    priorProbabilities[idx] += 1

#print(priorProbabilities)

priorCounts = priorProbabilities.copy()

def normalize(array):
    sum = 0

    for i in range(len(array)):
        sum += array[i]

    for i in range(len(array)):
        array[i] = array[i]/sum

    return array

priorProbabilitiesNormalized = normalize(priorProbabilities)
#print(priorCounts)
print("Prior probability")
print(priorProbabilitiesNormalized)



# transition probability calculation

transitionProbabilities = np.zeros((numberOfStates, 2))
#print(transitionProbabilities)
#

def subStringMatch(fullString, subString):
    subLen = len(subString)
    fullLen = len(fullString)
    count = 0

    for i in range(fullLen - subLen + 1):
        #print(fullString[i:i+subLen])
        if(fullString[i:i+subLen] == subString):
            count += 1
    #print(count)
    return count


x = subStringMatch(trainSequence, "1001")
#print(x)

for i in range(numberOfStates):
    binSequence = np.binary_repr(i, n)

    binSequence0 = binSequence + '0'
    binSequence1 = binSequence + '1'
    #print(binSequence, binSequence0, binSequence1)

    binSequence0_count = subStringMatch(trainSequence, binSequence0)
    binSequence1_count = subStringMatch(trainSequence, binSequence1)

    #print(binSequence0_count, binSequence1_count)

    sum = binSequence0_count + binSequence1_count

    transitionProbabilities[i][0] = (binSequence0_count/sum)
    transitionProbabilities[i][1] = (binSequence1_count/sum)

print("Transition probability")
print(transitionProbabilities)


# observation

def noise(var, numbers=1):
    mu, sigma = 0, var # mean and standard deviation
    s = np.random.normal(mu, sigma, numbers)

    return s[0]

def getX(weights, binSeq):
    sum = 0

    for i in range(len(weights)):
        sum += (weights[i] * float(binSeq[i]))
    sum += noise(variance)

    return sum

x = getX(w, "101")
#print(x)

observations = []
#print(observations)

#print(priorCounts)
#
for i in range(numberOfStates):

    numbers = int(priorCounts[i])
    x = []
    binSequence = np.binary_repr(i, n)
    for j in range(numbers):
        x.append(getX(w, binSequence))

    observations.append(x)

observations = np.array(observations)

#print(observations)


observationMeans = np.zeros((numberOfStates, 1))
#print(observationMeans)


for i in range(numberOfStates):
    numbers = int(priorCounts[i])
    sum = 0

    for j in range(numbers):
        sum += observations[i][j]

    observationMeans[i] = sum/numbers

print("Observation Means")
print(observationMeans)





# test observations


testObservations = np.zeros((len(testSequence), 1))
#print(testObservations)


len(testSequence)
for i in range(n-1, len(testSequence)):
    #print(i)

    start = i - (n-1)
    end = i+1

    subSeq = testSequence[start:end]
    #print(subSeq)

    testObservations[i] = getX(w, subSeq)

#print(testObservations)



# Viterbi algorithm



def getNextSequence(seq):
    seq0 = seq + '0'
    seq1 = seq + '1'

    seq0 = seq0[1:]
    seq1 = seq1[1:]

    return (seq0, seq1)

def getPreviousSeq(seq):
    seq0 = '0' + seq
    seq1 = '1' + seq

    seq0 = seq0[:n]
    seq1 = seq1[:n]

    return (seq0, seq1)


def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

#print(transitionProbabilities)


def getTransitionProbability(seq1, seq2):

    idx = int(seq1, 2)
    idx2 = int(seq2[n-1])

    return transitionProbabilities[idx][idx2]


stateProbabilities = np.zeros((numberOfStates, len(testSequence)))
stateSequence = np.zeros((numberOfStates, len(testSequence)))
#print(stateProbabilities)


for i in range(n-1, len(testSequence)):

    for j in range(numberOfStates):

        x = testObservations[i]
        mean = observationMeans[j]
        sd = variance


        if i == n-1:
            #print(x, mean, sd)
            stateProbabilities[j][i] = priorProbabilities[j] * normpdf(x, mean, sd)
        else:
            current = np.binary_repr(j, n)
            prev0, prev1 = getPreviousSeq(current)

            #print(prev0, prev1)

            transitionProbPrev0 = getTransitionProbability(prev0, current)
            transitionProbPrev1 = getTransitionProbability(prev1, current)

            #print(current, transitionProbPrev0, transitionProbPrev1)

            selectedPath = prev0
            selectedTransitionProb = transitionProbPrev0

            if transitionProbPrev1 > transitionProbPrev0:
                selectedPath = prev1
                selectedTransitionProb = transitionProbPrev1

            selectedPath = int(selectedPath, 2)
            #print(selectedPath)

            #print(i, j)
            stateProbabilities[j][i] = selectedTransitionProb * normpdf(x, mean, sd)
            stateSequence[j][i] = selectedPath

            #cost0 = stateProbabilities[i-1][]



#print(stateProbabilities)
#print(stateSequence)

#print(stateProbabilities.shape)

testLength = len(testSequence);

finalStateProb = stateProbabilities[ : , testLength-1 ]
#print(finalStateProb)

maxFinalState = np.max(finalStateProb)
maxFinalState = np.argmax(finalStateProb)
#print(maxFinalState)

prevState = maxFinalState
finalStateSeq = []
finalStateSeq.append(prevState)
for i in range(testLength-1, n-1, -1):
    prevState = int(stateSequence[prevState][i])
    finalStateSeq.append(prevState)

#print(finalStateSeq)

binarySeq = []
binarySeq.append(np.binary_repr(maxFinalState, n))

for i in range(len(finalStateSeq)-1, -1, -1):
    binary = np.binary_repr(finalStateSeq[i], n)
    binarySeq.append(binary[n-1:n])

#print(binarySeq)

str1 = ''.join(binarySeq)
#print(str1)

count = 0
for i in range(testLength):
    a1 = testSequence[i]
    a2 = str1[i]
    if a1 == a2:
        count += 1

#print(count)
print("Accuracy")
print((count/testLength)*100)