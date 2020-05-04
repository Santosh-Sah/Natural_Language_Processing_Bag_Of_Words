# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:01:34 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importNaturalLanguageProcessingBagOfWordsDataset(naturalLanguageProcessingBagOfWordsDatasetFileName):
    
    naturalLanguageProcessingBagOfWordsDataset = pd.read_csv(naturalLanguageProcessingBagOfWordsDatasetFileName, delimiter = '\t', quoting = 3)
    
    return naturalLanguageProcessingBagOfWordsDataset

def cleanNaturalLanguageProcessingBagOfWordsDataset(naturalLanguageProcessingBagOfWordsDataset, nltkStopwords):
    
    bagOfWordsCorpus = []
    
    for i in range(0, 1000):
        
        bagOfWordsReview = re.sub('[^a-zA-Z]', ' ', naturalLanguageProcessingBagOfWordsDataset['Review'][i])
        
        bagOfWordsReview = bagOfWordsReview.lower()
        
        bagOfWordsReview = bagOfWordsReview.split()
        
        bagOfWordsPorterStemmer = PorterStemmer()
        
        bagOfWordsReview = [bagOfWordsPorterStemmer.stem(word) for word in bagOfWordsReview if not word in set(nltkStopwords.words('english'))]
        
        bagOfWordsReview = ' '.join(bagOfWordsReview)
        
        bagOfWordsCorpus.append(bagOfWordsReview)
    
    return bagOfWordsCorpus


def createNaturalLanguageProcessingBagOfWordsModel(naturalLanguageProcessingBagOfWordsDataset, bagOfWordsCorpus):
    
    bagOfWordsCountVectorizer = CountVectorizer(max_features = 1500)
    bagOfWordsCountVectorizer.fit(bagOfWordsCorpus)
    
    X = bagOfWordsCountVectorizer.transform(bagOfWordsCorpus).toarray()
    y = naturalLanguageProcessingBagOfWordsDataset.iloc[:, 1].values
    
    return bagOfWordsCountVectorizer, X, y

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save NaiveByesClassificationModel as a pickle file.
"""
def saveNaiveByesClassificationModel(naiveByesClassificationModel):
    
    #Write NaiveByesClassificationModel as a picke file
    with open("NaiveByesClassificationModel.pkl",'wb') as NaiveByesClassificationModel_Pickle:
        pickle.dump(naiveByesClassificationModel, NaiveByesClassificationModel_Pickle, protocol = 2)

"""
read NaiveByesClassificationModel from pickle file
"""
def readNaiveByesClassificationModel():
    
    #load NaiveByesClassificationModel model
    with open("NaiveByesClassificationModel.pkl","rb") as NaiveByesClassificationModel:
        naiveByesClassificationModel = pickle.load(NaiveByesClassificationModel)
    
    return naiveByesClassificationModel

"""
read X_train from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveNaturalLanguageProcessingBagOfWordsYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred

"""
save bagOfWordsCountVectorizer as a pickle file
"""

def saveNaturalLanguageProcessingBagOfWordsCountVectorizer(countVectorizer):
    
    #Write CountVectorizer in a picke file
    with open("CountVectorizer.pkl",'wb') as countVectorizer_Pickle:
        pickle.dump(countVectorizer, countVectorizer_Pickle, protocol = 2)

"""
read bagOfWordsCountVectorizer from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsCountVectorizer():
    
    #load bagOfWordsCountVectorizer
    with open("CountVectorizer.pkl","rb") as CountVectorizer_pickle:
        countVectorizer = pickle.load(CountVectorizer_pickle)
    
    return countVectorizer