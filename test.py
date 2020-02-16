#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:38:01 2019

@author: hp
"""

import pandas as pd
import csv
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
import numpy as np
#import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def maxminposition(A, n):
   # inbuilt function to find the position of minimum 
   minposition = A.index(min(A))
   # inbuilt function to find the position of maximum 
   maxposition = A.index(max(A)) 
   return maxposition + 1 , minposition + 1

##  TAKE INPUT IN REVIEW IN SECOND INDEX [1]

review=["thank you","this hotel is good"]

#######  loading the actual representation of data and labels and vocabulary


vocabulary=np.load("unique_words.npy")
data=np.load("complete_process_data.npy")
label=np.load("complete_process_label.npy")

####   making TERM FREQUENCY + INVERSE DOCUMENT FREQUENCY OF INPUT DATA


vectorizer = TfidfVectorizer(vocabulary=vocabulary)
matrix=vectorizer.fit_transform(review).todense()

####   APPLYING COSINE SIMILARITY

xtrain=pd.DataFrame(data)
xtest=pd.DataFrame(matrix)
x=cosine_similarity(xtrain,xtest)
x=pd.DataFrame(x)
y_train=label

#             suppose value of k = 3
k=3

tempdf=x.sort_values(by=[1], axis=0, ascending =False, na_position= 'first')
val1=tempdf.index.tolist()
label=y_train[val1[0]]
value=x.iloc[val1[0],1]
    #print("label",label)
    #print("value",value)
    
storage=np.zeros(k)
values = np.zeros(k)
for i in range(0,k):
    storage[i]=y_train[val1[i]]
    values[i]=x.iloc[val1[i],1]
count=[0,0,0,0]

for i in range(0,k):
    if storage[i]==0:
        count[0]+=1
    elif storage[i]==1:
        count[1]+=1
    elif storage[i]==2:
        count[2]+=1
    elif storage[i]==3:
        count[3]+=1
max_pos,min_pos = maxminposition(count,4)


if max_pos-1==0:
    print("this review is  truthful and positive")
elif max_pos-1==1:
    print("this review is  truthful and negative")
elif max_pos-1==2:
    print("this review is  deceptive and positive")
elif max_pos-1==3:
    print("this review is  deceptive and negative")
        