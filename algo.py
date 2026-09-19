import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.metrics import accuracy_score

print('libraries imported')




i=load_iris()
df=pd.DataFrame(i.data,columns=i.feature_names)

df['Species']=i.target
print(" ")
print(df.head())
print(' ')



z=np.abs(zscore(df))
dfn=df[(z<3).all(axis=1)]
print('old',df.shape,'new',dfn.shape)




x=dfn.drop('Species',axis=1)
y=dfn['Species']



print(' ')

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.30,random_state=46)

print('xtrain.shape',xtrain.shape,'\nytrain.shape',ytrain.shape)
print(' ')




svc=SVC()
svc.fit(xtrain,ytrain)
print("Accuracy Score",accuracy_score(ytest,svc.predict(xtest)))


import joblib as j 
j.dump(svc,'svc.pkl')

print('model saved')