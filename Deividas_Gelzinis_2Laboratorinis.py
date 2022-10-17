from random import uniform
import numpy as np
irisResult = []
irisData = []
vezioResult = []
vezioData = []
weights = []
bias = 0

epochs = 10
learningRate = 0.001

def irisuSkaitymas():
    with open('iris.data', 'r') as f:
        i = 0
        for line in f.read().split("\n"):
            isWrong = 0
            irisTemp= []
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
            for x in line.split(","):
                if isfirst:
                    isfirst = 0
                else:
                    vezioTemp.append(float(x))
                if x == '?' or not x:
                    wrongData = 1
            if not wrongData:
                vezioData.append(vezioTemp)
        
def savingIrisResults():
    for data in irisData:
        irisResult.append(data.pop())

def savingVezioResults():
    for data in vezioData:
        newRes = 0
        if data.pop() == '2':
            newRes = 0
        else:
            newRes = 1
        vezioResult.append(newRes)
        
def generateWeights(dataChoice):
    bias = round(uniform(0, 1), 5)
    if dataChoice == '1':
        for _ in range(4):
            weights.append(round(uniform(0, 1), 5))
    else:
        for _ in range(9):
            weights.append(round(uniform(0, 1), 5))

def slenkstinisActivation(result, expectedResult):
    if expectedResult == 1 and result>=0:
        return 1
    elif expectedResult == 0 and result<0:
        return 1
    else:
        return 0

def sigmoidinisActivation(result, expectedResult):
    sigmoid = 1/(1+np.exp(-result))
    if expectedResult == 1 and sigmoid>=0.5:
        return 1
    elif expectedResult == 0 and sigmoid<0.5:
        return 1
    else:
        return 0

def neuronLearning(data, results, activationChoice):
    for _ in range(epochs):
        for i in data:
            a = np.dot(data[1], weights)
            if activationChoice == '1':
                slenkstinisActivation(a, results[i])
            elif activationChoice == '2':
                sigmoidinisActivation(a, results[i])


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
        neuronLearning(data, results, activationChoice)
    else:
        print('Wrong choice selection')

main()
