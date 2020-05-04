# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:27:01 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from NaturalLanguageProcessingBagOfWordsUtils import (readNaturalLanguageProcessingBagOfWordsYTest, readNaturalLanguageProcessingBagOfWordsYPred)

"""

calculating NaturalLanguageProcessingBagOfWords confussion matrix

"""
def testNaturalLanguageProcessingBagOfWordsConfussionMatrix():
    
    y_test = readNaturalLanguageProcessingBagOfWordsYTest()
    y_pred = readNaturalLanguageProcessingBagOfWordsYPred()
    
    naturalLanguageProcessingBagOfWordsConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(naturalLanguageProcessingBagOfWordsConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[55 42]
    [12 91]]
    
    """
"""
calculating accuracy score

"""

def testNaturalLanguageProcessingBagOfWordsAccuracy():
    
    y_test = readNaturalLanguageProcessingBagOfWordsYTest()
    y_pred = readNaturalLanguageProcessingBagOfWordsYPred()
    
    naturalLanguageProcessingBagOfWordsAccuracy = accuracy_score(y_test, y_pred)
    
    print(naturalLanguageProcessingBagOfWordsAccuracy) #.73%

"""
calculating classification report

"""

def testNaturalLanguageProcessingBagOfWordsClassificationReport():
    
    y_test = readNaturalLanguageProcessingBagOfWordsYTest()
    y_pred = readNaturalLanguageProcessingBagOfWordsYPred()
    
    naturalLanguageProcessingBagOfWordsClassificationReport = classification_report(y_test, y_pred)
    
    print(naturalLanguageProcessingBagOfWordsClassificationReport)
    
    """
             precision    recall  f1-score   support

          0       0.82      0.57      0.67        97
          1       0.68      0.88      0.77       103

avg / total       0.75      0.73      0.72       200
    """
    
if __name__ == "__main__":
    #testNaturalLanguageProcessingBagOfWordsConfussionMatrix()
    #testNaturalLanguageProcessingBagOfWordsAccuracy()
    testNaturalLanguageProcessingBagOfWordsClassificationReport()