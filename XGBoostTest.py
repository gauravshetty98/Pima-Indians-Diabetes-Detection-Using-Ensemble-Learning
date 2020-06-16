
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as AS
import numpy as np
import pandas as pd
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
import xgboost as xgb
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

dataset = pd.read_csv('newPimaIndians.csv')
groups=dataset.groupby('outcome')
field=['glucose','blood pressure','bmi','pedigree','age']


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
data=[[0,0,0,0,0]]
df=pd.DataFrame(data,columns=['feats','depth','split','max_leaf','acc'])
for feats in range(2, 7):
    for dept in range(2, 6):
        acc = 0
        for split in range(5,40,5):
            for leaf in range(7,10):
                for i in range(20):
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
                    classifier= GradientBoostingClassifier(min_samples_split=split,max_depth=dept,max_features=feats,max_leaf_nodes=leaf)
                    classifier.fit(X_train, Y_train)
                    res = classifier.score(X_test, Y_test)
                    acc = acc + res
                acc = acc / 20    
                print('feats:', feats, 'Depth:', dept,'split:',split,'max_leaf',leaf, 'acc:', acc*100)
                df=df.append({'feats':feats,'depth':dept,'split':split,'max_leaf':leaf,'acc':acc},ignore_index=True)
df.to_csv('xgboost.csv', sep=',')