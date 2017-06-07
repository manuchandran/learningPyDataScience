# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 00:12:36 2017

@author: Manu
"""
#IMPORT LIBRARIES 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn import svm 

#IMPORT DATA 
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")






