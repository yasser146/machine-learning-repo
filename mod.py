import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.tree import DecisionTreeClassifier


import seaborn as sns 
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.model_selection import GridSearchCV
data =pd.read_csv('data.csv')
data.info()
data.isna().sum()
data.dropna(axis=0 , inplace=True)
data.isna().sum()
data.hist(bins=30 , figsize=(14,16))
plt.show()
sns.boxplot(x=data['oldpeak'])
i = data[((data.oldpeak >= 6 ))].index
data.drop(i, inplace=True)
sns.boxplot(x=data['oldpeak'])
rcParams['figure.figsize'] = 10,15
rcParams["figure.dpi"]= 100
plt.matshow(data.corr())
plt.yticks(np.arange(data.shape[1]), data.columns)
plt.xticks(np.arange(data.shape[1]), data.columns)
plt.colorbar()
data['target'].unique()
rcParams['figure.figsize'] = 6,6
plt.bar(data['target'].unique(), data['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
y = data['target']
X = data.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =4)
Hyper_paramters={'criterion':['gini', 'entropy'],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
Tree_Gridsearch_paramters=GridSearchCV(DecisionTreeClassifier(),Hyper_paramters,scoring='roc_auc',n_jobs=-1,cv=10,verbose=1)
Tree_crossvalidation=Tree_Gridsearch_paramters.fit(X_train,y_train)


print ("The best paramter combination is ")
print(Tree_crossvalidation.best_params_) 
Final_Model=Tree_crossvalidation.best_estimator_ 
print("The best AUC score was ")
print(Tree_crossvalidation.best_score_)  
importances=Tree_crossvalidation.best_estimator_.feature_importances_
importances

Names=list(X_train.columns.values)
for f in range(X_train.shape[1]):
    print((Names[f],  importances[[f]]))

    y_pred = Final_Model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    cm1 = confusion_matrix(y_test, y_pred)
    print(cm1)
    print(classification_report(y_test, y_pred, target_names=["Heart disease", "No Heart Disears"]))
    
    #Calculate sensitivity and specificity
    
    total1=sum(sum(cm1))
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)
    
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )
    
    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity1)

import pickle
pickle.dump(Final_Model , open('model.pkl' , 'wb'))
model=pickle.load(open('model.pkl','rb'))





