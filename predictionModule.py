'''
Created on 15-Dec-2018

@author: Teerta shetty
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as AS
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier


dataset = pd.read_csv('pima-indians-diabetes.csv')
newDataset = pd.read_csv('pima-indians-diabetes.csv')
groups=dataset.groupby('outcome')
field=['glucose','blood pressure','skin thickness','insulin','bmi','pedigree','age']


#this for loop is for adding missing values
for f in field:
    temp=groups[f].median()
    for i in range(0,768):
        if (dataset.loc[i,f]==0) & (dataset.loc[i,'outcome']==0):
            dataset.loc[i,f]=temp[0]
        if (dataset.loc[i,f]==0) & (dataset.loc[i,'outcome']==1):
            dataset.loc[i,f]=temp[1]


dataset = dataset.values
X = dataset[:,0:len(dataset[0]) -1]
Y = dataset[:, (len(dataset[0])-1)]

#this is for decision tree
#Store the values in these variables according to the classification results. These values will be passed as parameters to the classification algorithms.
feats=4
dept=6
split=20
leaf=4


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
#select one classifier at a time
#classifier = RandomForestClassifier(max_features=feats, max_depth=dept,min_samples_split=split,criterion="entropy",max_leaf_nodes=leaf)
classifier = DecisionTreeClassifier(max_depth=dept, max_features=feats,min_samples_split=split,splitter="best",criterion="entropy",max_leaf_nodes=leaf)
#classifier=AdaBoostClassifier(DecisionTreeClassifier(max_depth=dept, max_features=feats,min_samples_split=split,splitter="best",criterion="entropy",max_leaf_nodes=leaf),learning_rate=1.0)                    
#classifier= GradientBoostingClassifier(min_samples_split=split,max_depth=dept,max_features=feats,max_leaf_nodes=leaf)
classifier.fit(X_train, Y_train)
res=classifier.predict(X_test)
data=[9,9]

# Rather than doing the long process given below you can simply use classifier.score() method to get the accuracy of the classifier.
# The following steps showed me all the incorrect results so that I could find any patterns in the incorrect data if any.
correctDF=pd.DataFrame()
incorrectDF=pd.DataFrame()


for i in range(0,231):
    if (Y_test[i]==res[i]):
        correctDF=correctDF.append(newDataset.iloc[i],ignore_index=True)
    else:
        incorrectDF=incorrectDF.append(newDataset.iloc[i],ignore_index=True)


correctDF=correctDF.reindex(newDataset.columns,axis=1)     
incorrectDF=incorrectDF.reindex(newDataset.columns,axis=1)     
print('correct results',correctDF)
print('incorrect results',incorrectDF)

     
correctDF.to_csv('rf_correct_full_class_4.csv',sep=',')
incorrectDF.to_csv('rf_incorrect_full_class_4.csv',sep=',')