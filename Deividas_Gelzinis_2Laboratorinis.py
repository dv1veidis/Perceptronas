import random
from random import uniform, shuffle
import numpy as np
import matplotlib.pyplot as plt

# Aprašomi naudojami masyvai
irisResult = []
irisData = []
vezioResult = []
vezioData = []
weights = []
learningSet = []
learningSetResults = []
learningAccuracy = []
learningErrors = []
testAccuracy = []
testErrors = []

epochs = 100
learningRate = 0.001

# Nuskaito ir sutvarko Irisu duomenis
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

# Nuskaito ir sutvarko Vezio duomenis
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
        
# Išsaugo Irisu klases į atskira masyva
def savingIrisResults():
    shuffle(irisData)
    print("Irisų duomenu aibių skaičius: "+str(len(irisData)))
    for data in irisData:
        irisResult.append(data.pop())

# Išsaugo Vežio klases į atskira masyva
def savingVezioResults():
    shuffle(vezioData)
    print("Vėžio duomenu aibių skaičius: " + str(len(vezioData)))
    for data in vezioData:
        newRes = 0
        if data.pop() == 2:
            newRes = 0
        else:
            newRes = 1
        vezioResult.append(newRes)
        
# Sugeneruojami pradiniai svoriai naudojant skaicius nuo 0 iki 1
def generateWeights(dataChoice):
    if dataChoice == '1':
        for _ in range(5):
            weights.append(uniform(0, 1))
    else:
        for _ in range(10):
            weights.append(uniform(0, 1))
    print(weights)

# Aprašoma slenkstine aktivacija
def slenkstinisActivation(result):
    if result>=0:
        return 1
    else:
        return 0

# Aprašoma sigmoidine aktivacija apvalinama nuo 0.7 ir 0.3
def sigmoidinisActivation(result):
    sigmoid = 1/(1+np.exp(-result))
    if sigmoid>=0.7:
        return 1
    elif sigmoid<=0.3:
        return 0
    else:
        return sigmoid

# Aprašoma adeline funkcija svoriu atnaujinimui
def adeline(w, t, y, x):
    for i in range(0, len(w)):
        w[i] = w[i] + learningRate * (t - y) * x[i]
    return w

# Nupiešiami tikslumo ir paklaidos duomenis pagal epocha
def drawResults(learningAccuracy, learningError):
    
    plt.plot(range(0,len(learningAccuracy)), learningAccuracy)
    # naming the x axis
    plt.xlabel('Epochos')
    # naming the y axis
    plt.ylabel('Tikslumas')
    
    plt.show()
    
    plt.plot(range(0,len(learningError)), learningError)
    # naming the x axis
    plt.xlabel('Epochos')
    # naming the y axis
    plt.ylabel('Paklaida')
    
    plt.show()

# Testuojamas neuronas su galutiniais svoriais
def neuronTest(weights, testSet, activationChoice, testSetResults):
    correct = 0
    error = 0
    for i in range(len(testSet)):    
        a = np.dot(testSet[i], weights)
        if activationChoice == '1':
            y = slenkstinisActivation(a)
            if y == testSetResults[i]:
                correct += 1
            error += pow(y - testSetResults[i], 2)
        elif activationChoice == '2':
            y = sigmoidinisActivation(a)
            if y == testSetResults[i]:
                correct += 1
            error += pow(y - testSetResults[i], 2)
    
    accuracy = 100 / len(testSet) * correct
    testAccuracy.append(accuracy)
    testErrors.append(error/2)
    print(testAccuracy)
    print(testErrors)

# Apmokomas neuronas naudojant adeline funkcija bei išskaidom duomenis santykiu 80/30
def neuronLearning(data, weights, results, activationChoice):
    # Skaidomi duomenis
    learnSetSize = int(len(data) * 80 / 100)
    learningSet = data[0:learnSetSize]
    learningSetResults = results[0:learnSetSize]
    testSet = data[learnSetSize:len(data)]
    testSetResults = results[learnSetSize:len(results)]
    # Pradedamos epochos
    for _ in range(epochs):
        correct = 0
        error = 0
        # Vykdomos iteracijos
        for i in range(len(learningSet)):
            # Atlieka 2 vektoriu apjungima i bendra suma
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
                # Atlieka paklaidos skaičiavima pakeliant gauta rezulta kvadratu
                error += pow((1/(1+np.exp(-a))) - learningSetResults[i], 2)
        # Aprašo epochos tiksluma
        learningAccuracy.append(100 / len(learningSet) * correct)
        # Aprašo epochos paklaida
        learningErrors.append(error / 2)
    drawResults(learningAccuracy, learningErrors)
    neuronTest(weights, testSet, activationChoice, testSetResults)

# Pagrindine veikimo funkcija
def main():
    # Pasirenkame norima duomenu rinkini
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
    # Pasirenkame norima aktivacijos funkcija
    activationChoice = input("Choose Slenkstine(1) or Sigmoidine(2) Activation:")
    if dataChoice == '1' or dataChoice == '2':
        neuronLearning(data, weights, results, activationChoice)
    else:
        print('Wrong choice selection')

main()
