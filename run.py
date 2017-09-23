# -*- coding: utf-8 -*-

from numpy import *
from os import listdir
from PIL import Image
from train import txt2vector
from math import sqrt

def img2vector(img):
    data = zeros((32, 32))
    px = img.load()
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            data[r, c] = px[c, r][0]
    return data.flatten()


def normalize(vector, maxValue):
    return [i / maxValue for i in vector]


def loadTrainedImages(path):
    files = listdir(path)
    images = [Image.open(path + "/" + f) for f in files]
    nums = [int(f.split('.')[0]) for f in files]
    data = zeros((10, 1024))
    for idx, num in enumerate(nums):
        imgData = img2vector(images[idx])
        data[num] = normalize(imgData, 255)
    return data


def selectRandomFilesFrom(path, count):
    files = listdir(path)
    indexes = random.choice(len(files), size=count).tolist()
    selectedPaths =  [path + "/" + files[i] for i in indexes]
    vectors = [txt2vector(path) for path in selectedPaths]
    nums = [int(p.split('/')[2].split('_')[0]) for p in selectedPaths]
    inputs = [[],[],[],[],[],[],[],[],[],[]]
    for i, num in enumerate(nums):
        arr = inputs[num]
        arr.append(vectors[i])
    return array(inputs)


def measureEuclideanDistance(array1, array2):
    distances = [val - array2[i] for i, val in enumerate(array1)]
    distances2 = [i * i for i in distances]
    return sqrt(sum(distances2))


def getIndexOfClosest(input, trainedData):
    distances = [measureEuclideanDistance(input, data) for data in trainedData]
    indexResult = 0;
    result = distances[indexResult]
    for i, d in enumerate(distances):
        if result > d:
            result = d
            indexResult = i
    return indexResult


# ----------------------  THE MAIN LOGIC  ----------------------------------

trainedData = loadTrainedImages('digits/trained')
inputCount = 200
inputs = selectRandomFilesFrom('digits/testDigits', inputCount)
mistakesCount = [0] * 10
mistakes = [[],[],[],[],[],[],[],[],[],[]]
for num, inputsForNum in enumerate(inputs):
    for input in inputsForNum:
        guess = getIndexOfClosest(input, trainedData)
        if guess != num:
            mistakesCount[num] += 1
            mistakes[num].append(guess)
    accuracy = (1 - mistakesCount[num] / len(inputsForNum)) * 100;
    print("Num #%i : %i%% accuracy" % (num, accuracy))

print("\n%i/%i mistakes were found." % (sum(mistakesCount), inputCount))
print("Total Accuracy : " + str((1 - (sum(mistakesCount) / inputCount)) * 100) + " %\n")

print("MISTAKES:" + "\n".join([str(i) + " : " + str(arr) for i, arr in enumerate(mistakes)]))