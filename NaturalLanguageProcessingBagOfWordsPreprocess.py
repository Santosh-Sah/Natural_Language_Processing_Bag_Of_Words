# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:22:06 2020

@author: Santosh Sah
"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split

from NaturalLanguageProcessingBagOfWordsUtils import (importNaturalLanguageProcessingBagOfWordsDataset, saveTrainingAndTestingDataset,
                                                      cleanNaturalLanguageProcessingBagOfWordsDataset, createNaturalLanguageProcessingBagOfWordsModel,
                                                      saveNaturalLanguageProcessingBagOfWordsCountVectorizer)

def preprocess():
    
    naturalLanguageProcessingBagOfWordsDataset = importNaturalLanguageProcessingBagOfWordsDataset("Restaurant_Reviews.tsv")
    
    naturalLanguageProcessingBagOfWordsCorpus = cleanNaturalLanguageProcessingBagOfWordsDataset(naturalLanguageProcessingBagOfWordsDataset, stopwords)
    
    bagOfWordsCountVectorizer, X, y = createNaturalLanguageProcessingBagOfWordsModel(naturalLanguageProcessingBagOfWordsDataset, naturalLanguageProcessingBagOfWordsCorpus)
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    
    saveNaturalLanguageProcessingBagOfWordsCountVectorizer(bagOfWordsCountVectorizer)
    

if __name__ == "__main__":
    preprocess()