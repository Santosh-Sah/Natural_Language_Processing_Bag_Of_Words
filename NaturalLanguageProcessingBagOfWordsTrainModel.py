# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:17:32 2020

@author: Santosh Sah
"""

from sklearn.naive_bayes import GaussianNB
from NaturalLanguageProcessingBagOfWordsUtils import (saveNaiveByesClassificationModel, readNaturalLanguageProcessingBagOfWordsXTrain, 
                                                      readNaturalLanguageProcessingBagOfWordsYTrain)

"""
Train NaiveByesClassification model 
"""
def trainNaiveByesClassificationModel():
    
    X_train = readNaturalLanguageProcessingBagOfWordsXTrain()
    y_train = readNaturalLanguageProcessingBagOfWordsYTrain()
    
    naiveByesClassification = GaussianNB()
    naiveByesClassification.fit(X_train, y_train)
    
    saveNaiveByesClassificationModel(naiveByesClassification)

if __name__ == "__main__":
    trainNaiveByesClassificationModel()