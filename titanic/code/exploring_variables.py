# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:51:54 2017
# For PYTHON 3.6 
@author: Manu
"""

"""
Data Dictionary
Variable	Definition	Key
survival 	Survival 	0 = No, 1 = Yes
pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
sex 	Sex 	
Age 	Age in years 	
sibsp 	# of siblings / spouses aboard the Titanic 	
parch 	# of parents / children aboard the Titanic 	
ticket 	Ticket number 	
fare 	Passenger fare 	
cabin 	Cabin number 	
embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

SOURCE : https://www.kaggle.com/headsortails/pytanic 
"""


# LOAD LIBRARIES 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
#LOAD DATA

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

#PRILIM DATA UNDERSTANDING , PRINT ETC
train.head(8)   # IF YOU RUN CODE THIS WONT PRINT
print(train.describe())

print(train.isnull().sum()) # SO AND SO FIELDS HAVE SO AND SO NULLS
print(test.info())

surv        = train[train['Survived'] == 1]
nosurv      = train[train['Survived'] == 0]
surv_col    = "blue"
nosurv_col  = "red"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"\
      %(len(surv), len(surv)/len(train)*100.0,\
        len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train) ))
 # WHAT IS 1.*len() do here ? 

"""
#PLOT VARIABLES 
plt.figure(figsize = (12,10))
foo = sns.heatmap(train.drop('PassengerId', axis =1).corr(), vmax = 0.6, square = True, annot = True)

cols = ['Survived','Pclass','Age', 'SibSp', 'Parch', 'Fare']
g = sns.pairplot(data = train.dropna(), vars = cols, size = 1.5, hue = 'Survived', palette = [nosurv_col, surv_col])
"""

#WHY DOESNT THE FOLLOWING WORK 
fig, ax1 = plt.subplots() # USE SAME AXIS FOR PLOTTING ONE TO OF THE OTHER fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
sns.distplot(surv['Age'].dropna().values, bins = range(0,81,1), ax = ax1, kde = False, color = "blue")
sns.distplot(nosurv['Age'].dropna().values, bins = range(0,81,1), ax = ax2, kde = False, color = "red")

tab = pd.crosstab(train['Pclass'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)

#CREATE TRAIN AND TEST DATASET 
    #PREPARE DATA FOR NUMERICAL ANALYSIS
train["Sex"] = train["Sex"].astype("category")
train["Sex"].cat.categories = [0,1]
train["Sex"] = train["Sex"].astype("int")
train["Embarked"] = train["Embarked"].astype("category")
train["Embarked"].cat.categories = [0,1,2]
train["Embarked"] = train["Embarked"].astype("int")

test["Sex"] = test["Sex"].astype("category")
test["Sex"].cat.categories = [0,1]
test["Sex"] = test["Sex"].astype("int")
test["Embarked"] = test["Embarked"].astype("category")
test["Embarked"].cat.categories = [0,1,2]
test["Embarked"] = test["Embarked"].astype("int")

train.loc[:,["Sex","Embarked"]].head()

training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
     %(train.shape[0],training.shape[0],testing.shape[0]))

cols  = ['Sex','Pclass','Parch','SibSp','Embarked']
tcols = np.append(['Survived'],cols)

df = training.loc[:,tcols].dropna()
X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:,tcols].dropna()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['Survived']])

# TESTING LRegression 
clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = cross_val_score(clf_log, X, y, cv=5).mean()
print(score_log)

# TESTING SVM 
clf_svm = svm.SVC(
    class_weight='balanced'
    )
clf_svm.fit(X, y)
score_svm = cross_val_score(clf_svm, X, y, cv=5).mean()
print(score_svm)

#PLOT SVM BOUNDARY WITH DIFFERENT FEATURE PAIR. 
