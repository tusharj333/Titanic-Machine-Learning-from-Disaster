# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:21:58 2018

@author: Tushar
"""
import numpy as np
import pandas as pd

#%%
titanic_train = pd.read_csv("train.csv")
df = pd.DataFrame.copy(titanic_train)

#<<<<<<<<<<<< Train dataset >>>>>>>>>>>>
print(df.head())
print(df.shape)  #(891, 12)
print(df.describe(include = "all"))
print(df.isnull().sum()) 

#Age, Cabin, Embarked has missing values in train data set

#<<<<<<<<<<<< Test dataset >>>>>>>>>>>>
titanic_test = pd.read_csv("test.csv")
df_test = pd.DataFrame.copy(titanic_test)

print(df_test.head())
print(df_test.shape)  #(418, 11)
print(df_test.describe(include = "all"))
print(df_test.isnull().sum()) 

#Age & Cabin has missing values in data set

#%% Treating Missing values

#<<<<<<<<<<<< Train dataset >>>>>>>>>>>>

# Handling missing values of Cabin
# As there are lots of NAs(687 out of 891) in Cabin column we will drop that Column itself

df = df.drop("Cabin", axis = 1)

# Embarked has 2 missing values -->> fill by mode 

df['Embarked'].fillna(df["Embarked"].describe(include = "all")[2], inplace = True)
 
# Age has 177 missing values -- > fill by mean 

df["Age"].fillna(df.Age.mean(), inplace = True)

# Check for missing values again
print(df.isnull().sum()) 

# No missing values

#<<<<<<<<<<<< Test dataset >>>>>>>>>>>>

# Handling missing values of Cabin
# As there are lots of NAs(327 out of 418) in Cabin column we will drop that Column itself

df_test = df_test.drop("Cabin", axis = 1)
# Age has 86 missing values -- > fill by mean 

df_test["Age"].fillna(df_test.Age.mean(), inplace = True)

print(df_test.isnull().sum()) 

#%% Converting Categorical data to numerical data 
from sklearn import preprocessing

le = {}

#<<<<<<<<<<<< Train dataset >>>>>>>>>>>>

colname = ["Sex","Embarked"]


for feature in colname:
    le[feature] = preprocessing.LabelEncoder()  ## assigns numbers to the different levels in categorical features

for feature in colname:
    df[feature] = le[feature].fit_transform(df.__getattr__(feature))

#<<<<<<<<<<<< Test dataset >>>>>>>>>>>>
colname_1 = ["Sex","Embarked"]

for feature in colname_1:
    le[feature] = preprocessing.LabelEncoder()  ## assigns numbers to the different levels in categorical features

for feature in colname_1:
    df_test[feature] = le[feature].fit_transform(df_test.__getattr__(feature))
 

#%% Removing non significant features 
# keeping P-Class,Sex,Age,Embarked and removing all other remaining features
#<<<<<<<<<<<< Train dataset >>>>>>>>>>>>

#Split X and Y for training 

X = df[["Pclass","Sex","Age", "Embarked"]]
Y = df[["Survived"]]

print(X.head())
print(Y.head())
## All vlaues in X and Y are Numerical now

#<<<<<<<<<<<< Test dataset >>>>>>>>>>>>
X_test = df_test[["Pclass","Sex","Age", "Embarked"]]
print(X_test.head())


#%%Scaling for train 
# As the ranges of different independent variable varies a lot, the model may be 
# biased towards some of the higher range independent variables. Distribution is uneven

#%% Model Builing 
# 1 - Logistic Regression 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
#fitting training data to the model
classifier.fit(X,Y)
Y_pred_LR = classifier.predict(X_test)

df_test["Survived_LR"] = Y_pred_LR

#%%
# 2 - SVM
from sklearn import svm 
svc_model = svm.SVC(kernel = 'rbf', C=1.0, gamma = 0.1)
svc_model.fit(X,Y)
Y_pred_SVM = svc_model.predict(X_test)

df_test["Survived_SVM"] = Y_pred_SVM

#%%
# 3 - Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X,Y)

Y_pred_RF = classifier.predict(X_test)

df_test["Survived_RF"] = Y_pred_RF


#%%
# 4  -  K-NN 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
#fitting training data to the model
classifier.fit(X,Y)
Y_pred_KNN = classifier.predict(X_test)

df_test["Survived_KNN"] = Y_pred_KNN

#%%

# Submission

# Logistic Regression
df_LR = df_test[["PassengerId", "Survived_LR"]]
df_LR.to_csv("df.csv") # Score : 0.76076

# SVM
df_SVM = df_test[["PassengerId", "Survived_SVM"]]

# Random Forest
df_RF = df_test[["PassengerId", "Survived_RF"]]

# K-NN
df_KNN = df_test[["PassengerId", "Survived_KNN"]]








