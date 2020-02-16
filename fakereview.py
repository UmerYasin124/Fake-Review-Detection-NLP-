# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:34:46 2019

@author: hp
"""

#kaggle data set

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


strng=""
deceptive = []
hotel = []
polarity=[]
source=[]
text=[]
tokens=[]
lengtha=0
okuniquewords=[]
withoutstop_words=[]
count=0
pol=np.zeros(1600)


        
        
df=pd.read_csv("deceptive-opinion.csv")
x_train=df["text"].values
y_train1=df["deceptive"].values
y_train2=df["polarity"].values

y_train=np.zeros(1600)

index=0
for i,j in zip(y_train1,y_train2):
    if i=="truthful" and j=="positive":
        y_train[index]=0
    elif i=="truthful" and j=="negative":
        y_train[index]=1
    elif i=="deceptive" and j=="positive":
        y_train[index]=2
    elif i=="deceptive" and j=="negative":
        y_train[index]=3
    index+=1
    
np.save("train.npy",x_train)
np.save("label.npy",y_train)
        
#print(len(deceptive))
#print(len(hotel))
#print(len(polarity))
#print(len(source))
#print(len(text))
#tokenization
tokenizer = RegexpTokenizer(r'\w+')
for line in x_train:
        tokens.append(tokenizer.tokenize(line))
#unique words
for i in range(len(x_train)):
    for word in tokens[i]:   
        if word not in okuniquewords:              
            okuniquewords.append(word)  

np.save("tokens.npy",tokens)
np.save("uniquewords.npy",okuniquewords)
#removal of stop words
engilsh_stopwords = set(stopwords.words('english'))
for word in okuniquewords: 
    if word not in engilsh_stopwords:
        withoutstop_words.append(word)
        
unique_words=np.array(withoutstop_words)
np.save("unique_words.npy",unique_words)

#vsm of tf-idf
vectorizer = TfidfVectorizer(vocabulary=withoutstop_words)
matrix=vectorizer.fit_transform(x_train).todense()
matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
#i save this because we cann some how use this later
#np.save("/home/rana/mypython/umer/complete_process_data.npy",matrix)
#np.save("/home/rana/mypython/umer/complete_process_label.npy",y_train)
#getting labels of matrix



#matrix.to_csv("/home/rana/mypython/umer/tfidf.csv",index=False)

    
X_train,Y_train=shuffle(matrix,y_train, random_state=2)

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.2,random_state=4)

xtrain=pd.DataFrame(X_train)
xtest=pd.DataFrame(Y_train)
x=cosine_similarity( xtrain,xtest.iloc[100,:])

print(xtest.iloc[100,:])

x=pd.DataFrame(x)



tempdf=x.sort_values(by=[0], axis=0, ascending =False, na_position= 'first')
val1=tempdf.index.tolist()
label=y_train[val1[0]]
value=x.iloc[val1[0],0]
print("label",label)
print("value",value)
print("actual value",y_test[100])



