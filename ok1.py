# -*- coding: utf-8 -*-
"""
Created on Fri May 10 02:47:05 2019

@author: hp
"""

import pandas as pd
import csv
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class datapickup2():
    def setup1(self,a):
        review=["thank you",a]

#######  loading the actual representation of data and labels and vocabulary


        vocabulary=np.load("umer/unique_words.npy")
        data=np.load("umer/complete_process_data.npy")
        label=np.load("umer/complete_process_label.npy")
        
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
                
        minposition = count.index(min(count))
   # inbuilt function to find the position of maximum 
        maxposition = count.index(max(count)) 
        maxposition += 1 
        minposition += 1
       
        if maxposition-1==0:
            return("this review is  truthful and positive")
        elif maxposition-1==1:
            return("this review is  truthful and negative")
        elif maxposition-1==2:
            return("this review is  deceptive and positive")
        elif maxposition-1==3:
            return("this review is  deceptive and negative")
        
                  
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

