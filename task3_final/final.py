#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:49:09 2020

@author: jenar
"""

#Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#%%
#Importing the dataset and creating dataframe

dataset=pd.read_csv("student-math.csv",sep=';')


#%%
#Adding 'total_grades' to the dataframe

dataset['final_grade'] = dataset['G1'] + dataset['G2'] + dataset['G3']
temp_data = dataset.copy()

#%%
#Encoding all binary values with 1 and 0 in the dataframe

encoder = LabelEncoder()
dataset['school'] = encoder.fit_transform(dataset['school'])
dataset['sex'] = encoder.fit_transform(dataset['sex'])
dataset['address'] = encoder.fit_transform(dataset['address'])
dataset['famsize'] = encoder.fit_transform(dataset['famsize'])
dataset['Pstatus'] = encoder.fit_transform(dataset['Pstatus'])
dataset['schoolsup'] = encoder.fit_transform(dataset['schoolsup'])
dataset['famsup'] = encoder.fit_transform(dataset['famsup'])
dataset['paid'] = encoder.fit_transform(dataset['paid'])
dataset['activities'] = encoder.fit_transform(dataset['activities'])
dataset['nursery'] = encoder.fit_transform(dataset['nursery'])
dataset['higher'] = encoder.fit_transform(dataset['higher'])
dataset['internet'] = encoder.fit_transform(dataset['internet'])
dataset['romantic'] = encoder.fit_transform(dataset['romantic'])

#Encoding all nominal values in the dataframe

dataset['Mjob'] = encoder.fit_transform(dataset['Mjob'])
dataset['Fjob'] = encoder.fit_transform(dataset['Fjob'])
dataset['reason'] = encoder.fit_transform(dataset['reason'])
dataset['guardian'] = encoder.fit_transform(dataset['guardian'])
transcol = ColumnTransformer([('encoder',OneHotEncoder(),['Mjob', 'Fjob', 'reason', 'guardian'])], remainder = 'passthrough')
dataset = np.array(transcol.fit_transform(dataset), dtype = np.int64)

#%%
#initializing X and y
y = dataset[:,-1]
X = dataset[:,:-2]


#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#%% part2

#linear regression model
from sklearn.linear_model import LinearRegression

regresor = LinearRegression()
regresor.fit(X_train, y_train)


#Checking the score  
train_score = regresor.score(X_train, y_train)
test_score = regresor.score(X_test, y_test)
#Predicting
y_pred = np.int64(regresor.predict(X_test))
prediction_score = regresor.score(X_test, y_pred)

print(f'Train Score: {train_score}' )  
print(f'Test Score: {test_score}')
print(f'Predict Score: {prediction_score}' )


#plotting between the true and predicted values of x_test
plt.scatter(y_test, y_pred, color='blue')
plt.title('plot between the true and predicted values of x_test')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show() 

#%%

import statsmodels.api as sm

def backwardElimination(x, sl):
	numVars = len(x[0])
	for i in range(0, numVars):
		regressor_OLS = sm.OLS(y, x).fit()
		maxVar = max(regressor_OLS.pvalues).astype(float)
		if maxVar > sl:
			for j in range(0, numVars - i):
				if (regressor_OLS.pvalues[j].astype(float) == maxVar):
					x = np.delete(x, j, 1)
	print(regressor_OLS.summary())
	return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]
X_Modeled = backwardElimination(X_opt, SL)



#%%

#Bar-plot that provide insight how the 'sex' of the students relate their 'final_grade'
sns.catplot(x='absences', y='final_grade', data=temp_data, kind='bar')
sns.catplot(x='G1', y='final_grade', data=temp_data, kind='bar')
sns.catplot(x='G2', y='final_grade', data=temp_data, kind='bar')
sns.catplot(x='age', y='final_grade', hue='sex', data=temp_data, kind='bar')
sns.catplot(x='studytime', y='final_grade', hue='sex', data=temp_data, kind='bar')


#%% Decision Tree

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state = 0)
regressor1.fit(X_train1, y_train1)


train_score1 = regresor.score(X_train1, y_train1)
test_score1 = regresor.score(X_test1, y_test1)
y_pred1 = np.int64(regresor.predict(X_test1))
prediction_score1 = regresor.score(X_test1, y_pred1)

print(f'Train Score: {train_score1}' )  
print(f'Test Score: {test_score1}')
print(f'Predict Score: {prediction_score1}' )

plt.scatter(y_test1, y_pred1, color='red')
plt.title('y_pred vs y_test (Decision Tree Regression)')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show() 

#%%  RandomForest

from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor2.fit(X_train2, y_train2)

train_score2= regresor.score(X_train2, y_train2)
test_score2= regresor.score(X_test2, y_test2)
y_pred2= np.int64(regresor.predict(X_test2))
prediction_score2= regresor.score(X_test2, y_pred2)

print(f'Train Score: {train_score2}' )  
print(f'Test Score: {test_score2}')
print(f'Predict Score: {prediction_score2}' )

plt.scatter(y_test2, y_pred2, color='red')
plt.title('y_pred vs y_test (Random_Forest_Regression)')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show() 