# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:06:11 2019

@author: Alex
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_excel("features.xlsx",sheet_name="MFCC")
#print(df)
X = df.drop('Class',axis=1)
#print(X)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

clf = SVC(C=5000,kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
