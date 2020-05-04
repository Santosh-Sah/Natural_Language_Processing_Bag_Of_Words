# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:15:53 2020

@author: Santosh Sah
"""
import re
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from NaturalLanguageProcessingBagOfWordsUtils import readNaiveByesClassificationModel, readNaturalLanguageProcessingBagOfWordsCountVectorizer

def predict():
    
    naiveByesClassification = readNaiveByesClassificationModel()
    
    naturalLanguageProcessingBagOfWordsCountVectorizer = readNaturalLanguageProcessingBagOfWordsCountVectorizer()
    
    #bagOfWordsReview = "This is a good restaurant. Food is best in the city"
    bagOfWordsReview = "This is a bad restaurant. Food is bad in the city"
    
    bagOfWordsReview = re.sub('[^a-zA-Z]', ' ', bagOfWordsReview)
        
    bagOfWordsReview = bagOfWordsReview.lower()
        
    bagOfWordsReview = bagOfWordsReview.split()
        
    bagOfWordsPorterStemmer = PorterStemmer()
        
    bagOfWordsReview = [bagOfWordsPorterStemmer.stem(word) for word in bagOfWordsReview if not word in set(stopwords.words('english'))]
        
    bagOfWordsReview = ' '.join(bagOfWordsReview)
    
    bagOfWordsCorpus = []
    
    bagOfWordsCorpus.append(bagOfWordsReview)
    
    newObservation = naturalLanguageProcessingBagOfWordsCountVectorizer.transform(bagOfWordsCorpus).toarray()
        
    predictedValue = naiveByesClassification.predict(newObservation)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()