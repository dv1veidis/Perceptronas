import random
from random import uniform, shuffle
import numpy as np

irisResult = []
irisData = []
vezioResult = []
vezioData = []
weights = []
learningSet = []
learningSetResults = []
testSet = []
testSetResults = []
learningAccuracy = []
learningErrors = []

epochs = 10
learningRate = 0.0001

def irisuSkaitymas():
    with open('iris.data', 'r') as f:
        i = 0
        for line in f.read().split("\n"):
            isWrong = 0
            irisTemp= []
            irisTemp.append(1)
            for x in line.split(","):
                if x == 'Iris-setosa' or not x:
                    isWrong = 1
                else:
                    if x == 'Iris-versicolor':
                        irisTemp.append(0)
                    elif x == 'Iris-virginica':
                        irisTemp.append(1)
                    else:
                        irisTemp.append(float(x))
            if not isWrong:
                irisData.append(irisTemp)
                i =i + 1
        
def vezioSkaitymas():
    with open('breast-cancer-wisconsin.data', 'r') as f:
        i = 0
        for line in f.read().split("\n"):
            isfirst = 1
            wrongData = 0
            vezioTemp = []
            vezioTemp.append(1)
            for x in line.split(","):
                if isfirst:
                    isfirst = 0
                else:
                    if x != '?':
                        vezioTemp.append(float(x))
                if x == '?' or not x:
                    wrongData = 1
            if not wrongData:
                vezioData.append(vezioTemp)
        
def savingIrisResults():
    shuffle(irisData)
    for data in irisData:
        irisResult.append(data.pop())

def savingVezioResults():
    shuffle(vezioData)
    for data in vezioData:
        newRes = 0
        if data.pop() == '2':
            newRes = 0
        else:
            newRes = 1
        vezioResult.append(newRes)
        
def generateWeights(dataChoice):
    if dataChoice == '1':
        for _ in range(5):
            weights.append(uniform(0, 1))
    else:
        for _ in range(10):
            weights.append(uniform(0, 1))
    print(weights)

def slenkstinisActivation(result):
    if result>=0:
        return 1
    else:
        return 0

def sigmoidinisActivation(result):
    sigmoid = 1/(1+np.exp(-result))
    if sigmoid>=0.99:
        return 1
    else:
        return 0


def adeline(w, t, y, x):
    for i in range(0, len(w)):
        w[i] = w[i] + learningRate * (t - y) * x[i]
    return w

def neuronLearning(data, weights, results, activationChoice):
    learnSetSize = int(len(data) * 80 / 100)
    learningSet = data[0:learnSetSize]
    learningSetResults = results[0:learnSetSize]
    testSet = data[learnSetSize:len(data)]
    testSetResults = results[learnSetSize:len(results)]
    for _ in range(epochs):
        correct = 0
        error = 0
        for i in range(len(learningSet)):
            a = np.dot(learningSet[i], weights)
            if activationChoice == '1':
                y = slenkstinisActivation(a)
                if y == learningSetResults[i]:
                    correct += 1
                else:
                    weights = adeline(weights, learningSetResults[i], y, learningSet[i])
                error += pow(y - learningSetResults[i], 2)
            elif activationChoice == '2':
                y = sigmoidinisActivation(a)
                if y == learningSetResults[i]:
                    correct += 1
                else:
                    weights = adeline(weights, learningSetResults[i], y, learningSet[i])
                    print("I got it wrong")
                error += pow(y - learningSetResults[i], 2)
        learningAccuracy.append(100 / len(learningSet) * correct)
        learningErrors.append(error / 2)
    # print(learningAccuracy)
    # print(learningErrors)


def main():
    dataChoice = input("Choose Irisu(1) or Vezio(2) data:")
    if dataChoice == "1":
        irisuSkaitymas()
        savingIrisResults()
        generateWeights(dataChoice)
        data = irisData
        results = irisResult
    elif dataChoice == "2":
        vezioSkaitymas()
        savingVezioResults()
        generateWeights(dataChoice)
        data = vezioData
        results = vezioResult
    else:
        print('Wrong choice selection')
    activationChoice = input("Choose Slenkstine(1) or Sigmoidine(2) Activation:")
    if dataChoice == '1' or dataChoice == '2':
        neuronLearning(data, weights, results, activationChoice)
    else:
        print('Wrong choice selection')

main()
