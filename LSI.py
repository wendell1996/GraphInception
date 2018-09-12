#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:47:09 2018

@author: wendellcoma
"""

import math
import numpy as np
import time
import re

class LSI:
    def stripNonChar(self,string):
        stripped = (c for c in string if 'a'<=c<='z' or 'A'<=c<='Z' or '0'<=c<='9' or c==' ')
        return ''.join(stripped)
    
    def PPMI(self,pa,pb,pab,len):
        PMI = math.log10((pab+1)/((pa*pb)+len))
        return max(0,PMI)

    def trainModel(self,path,window=7,method='PPMI'):
        print('training latent semantic...')
        start = time.time()
        self.wordList = []
        self.wordDic = {}
        with open(path,'r') as f:
            for line in f:
                context = line.split(' ')
        f.close()        
        for word in context:
            if word not in self.wordList:
                self.wordList.append(word)
                self.wordDic.update({word:len(self.wordList)-1})
        self.frequencyMat = np.zeros([len(self.wordList),len(context)])
        self.PPMIMatrix = np.zeros([len(self.wordList),len(context)])
        counter = 0
        tempWordList = []
        for i,word in enumerate(context):
            if i%100 == 0:
                print('word%d trained'%i)
                print(word)
            if counter <= window:
                counter += 1
                if counter == 1:
                    continue
            else:
                del tempWordList[0]
            tempWordList.append(word)
            for wordIndex in range(len(tempWordList)-1):
                self.frequencyMat[self.wordDic.get(tempWordList[wordIndex]),i] += 1
                self.frequencyMat[self.wordDic.get(word),i-window+wordIndex] += 1
        if method == 'PPMI':
            self.wordFrequency = np.sum(self.frequencyMat,axis=1)
            frequencySum = np.sum(self.wordFrequency)
            frequencySumVec = np.full(self.wordFrequency.shape,frequencySum)
            frequencySumMat = np.full(self.frequencyMat.shape,frequencySum)
            self.wordFrequency = np.divide(self.wordFrequency,frequencySumVec)
            print(self.wordFrequency)
            self.frequencyMat = np.divide(self.frequencyMat,frequencySumMat)
            print(self.frequencyMat)
            for i in range(self.frequencyMat.shape[0]):
                for j in range(self.frequencyMat.shape[1]):
                    self.PPMIMatrix[i][j] = self.PPMI(self.wordFrequency[i],self.wordFrequency[self.wordDic.get(context[j])],self.frequencyMat[i][j],len(self.wordList))
            self.wordEmbedding,self.S,self.contextEmbedding=np.linalg.svd(self.PPMIMatrix)
        elif method == 'Frequency':
            print('SVD decomposing...')
            self.wordEmbedding,self.S,self.contextEmbedding=np.linalg.svd(self.frequencyMat)
        duration = time.time() - start    
        print('LSI_trained!(%.3fsec)\n'%duration)
        return
    
    def getWordList(self):
        return self.wordList
    
    def getWordDic(self):
        return self.wordDic
    
    def getFrequencyMat(self):
        return self.frequencyMat

    def getPPMIMatrix(self):
        return self.PPMIMatrix
    
    def getWordEmbedding(self,k=300):
        if k <= self.wordEmbedding.shape[1]:
            return self.wordEmbedding[:,0:k]
        else:
            return self.wordEmbedding
    
    def getContextEmbedding(self,k=300):
        if k <= self.contextEmbedding.shape[0]:
            return self.contextEmbedding[0:k,:] 
        else:
            return self.contextEmbedding
    
    def getSingular(self,k=300):
        if k <= self.S.shape[0]:
            return self.S[k]
        else:
            return self.S

    def word2vec(self,word,k=300):
        if k <= self.wordEmbedding.shape[1]:
            return self.wordEmbedding[self.wordDic.get(word),0:k]
        else:
            return self.wordEmbedding[self.wordDic.get(word),:]
        