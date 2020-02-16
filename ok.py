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

class datapickup():
    def setup(self):
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
        
        return x_train
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

