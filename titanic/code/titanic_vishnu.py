# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD DATA

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

data.head()
print(data)

p = data['Pclass']
s = data['Survived']
s1=[y for x,y in zip(p,s) if x == 1]
s2=[y for x,y in zip(p,s) if x == 2]
s3=[y for x,y in zip(p,s) if x == 3]

class1_survival = sum(s1)/len(s1)*100
class2_survival = sum(s2)/len(s2)*100
class3_survival = sum(s3)/len(s3)*100
plt.figure(1)
ax=sns.barplot([1,2,3],[class1_survival,class2_survival,class3_survival])
ax.set(ylabel='Survival Chances', xlabel='Pclass')

age = data['Age']
age1 = [x for x,y in zip(age,s) if y == 1 and not np.isnan(x)]
plt.figure(2)
ax2=sns.distplot(age1, bins=60, kde=False, rug=True);
ax2.set(xlabel='Age', ylabel='No of passengers survided')


age0 = [x for x,y in zip(age,s) if y == 0 and not np.isnan(x)]
plt.figure(3)
ax2=sns.distplot(age0, bins=60, kde=False, rug=True);
ax2.set(xlabel='Age', ylabel='No of passengers died')