# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:51:54 2017

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
"""
# LOAD LIBRARIES 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD DATA

train = pd.read_csv("..//input//train.csv")
test  = pd.read_csv("..//input//test.csv")

#PRILIM DATA UNDERSTANDING , PRINT ETC
train.head(8)   # IF YOU RUN CODE THIS WONT PRINT
print(train.describe())

print(train.isnull().sum())
print(test.info())

surv        = train[train['Survived'] == 1]
nosurv      = train[train['Survived'] == 0]
surv_col    = "blue"
nosurv_col  = "red"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"\
      %(len(surv), len(surv)/len(train)*100.0,\
        len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train) ))
 # WHAT IS 1.*len() do here ? 

#PLOT VARIABLES 
plt.figure(figsize = (12,10))
foo = sns.heatmap(train.drop('PassengerId', axis =1).corr(), vmax = 0.6, square = True, annot = True)

cols = ['Survived','Pclass','Age', 'SibSp', 'Parch', 'Fare']
g = sns.pairplot(data = train.dropna(), vars = cols, size = 1.5, hue = 'Survived', palette = [nosurv_col, surv_col])


sns.distplot(surv['Age'].dropna().values, bins = range(0,81,1), ax = ax1, kde = False, color = "blue")
sns.distplot(nosurv['Age'].dropna().values, bins = range(0,81,1), ax = ax2, kde = False, color = "red")

#WHY DOESNT THE FOLLOWING WORK 
fig, ax1 = plt.subplots() # USE SAME AXIS FOR PLOTTING ONE TO OF THE OTHER fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
sns.distplot(surv['Age'].dropna().values, bins = range(0,81,1), ax = ax1, kde = False, color = "blue")
sns.distplot(nosurv['Age'].dropna().values, bins = range(0,81,1), ax = ax2, kde = False, color = "red")






