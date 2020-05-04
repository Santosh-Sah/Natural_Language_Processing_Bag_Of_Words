# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:47:27 2020

@author: Santosh Sah
"""

from NaturalLanguageProcessingBagOfWordsUtils import (readNaturalLanguageProcessingBagOfWordsXTest, readNaiveByesClassificationModel,
                                     saveNaturalLanguageProcessingBagOfWordsYPred)

"""
test the model on testing dataset
"""
def testNaiveByesClassificationModel():
    
    X_test = readNaturalLanguageProcessingBagOfWordsXTest()
    
    naiveByesClassificationModel = readNaiveByesClassificationModel()
    
    y_pred = naiveByesClassificationModel.predict(X_test)
    
    saveNaturalLanguageProcessingBagOfWordsYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testNaiveByesClassificationModel()