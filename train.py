# -*- coding: utf-8 -*-

from numpy import *
from os import listdir
from PIL import Image


def txt2vector(filename):
    returnVect = zeros(1024)
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[32 * i + j] = int(lineStr[j])
    return returnVect


def merge(vectorA, vectorB):
    for idx, val in enumerate(vectorB):
        vectorA[idx] += val


def normalizeVector(combinedVector, count, maxValue):
    normalized = zeros(len(combinedVector), int)
    for idx, val in enumerate(combinedVector):
        normalized[idx] = int(val / count * maxValue)
    return normalized


def getRawDataFromFiles(path):
    vectors = []
    for i in range(10):
        vectors.append([])
    fileList = listdir(path)
    for fileN in fileList:
        idx = int(fileN.split('_')[0])
        vectors[idx].append(txt2vector('%s/%s' % (path, fileN)))
    return vectors


def train(rawData):
    trainedVectors = []
    for  num, rawVectorsForNum in enumerate(rawData):
        trainedVectors.append(zeros(1024))
        for rawVect in rawVectorsForNum:
            merge(trainedVectors[num], rawVect)
        trainedVectors[num] = normalizeVector(
            trainedVectors[num], len(rawVectorsForNum), 255)
    return trainedVectors


def matrix2image(matrix):
    img = Image.new("RGB", (len(matrix), len(matrix[0])))
    px = img.load()
    for r, row in enumerate(matrix):
        for c, val in enumerate(matrix[r]):
            px[c, r] = (val, val, val)
    return img


def save(trainedData, path):
    for num, vector in enumerate(trainedData):
        img = matrix2image(vector.reshape(32, 32))
        fileName = '%i.png' % num
        img.save('%s/%s' % (path, fileName))
    return

def train():
    print('Reading Raw Data')
    rawData = getRawDataFromFiles('digits/testDigits')
    print('Merging Raw Data')
    trainedData = train(rawData)
    print('Saving Results')
    save(trainedData, 'digits/trained')
    print('Done')
