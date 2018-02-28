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
print(df.head())
print(df.shape)  #(891, 12)
print(df.describe(include = "all"))
print(df.isnull().sum()) 

#Age, Cabin, Embarked has missing values in data set


#%% Treating Missing values

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

#%%








