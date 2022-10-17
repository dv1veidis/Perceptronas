import numpy as np
irisResult = []
irisData = []
vezioResult = []
vezioData = []

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
                        irisTemp.append(x)
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
                    vezioTemp.append(x)
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
    print(vezioResult)
        


def main():
    irisuSkaitymas()
    savingIrisResults()
    vezioSkaitymas()
    savingVezioResults()
    
main()