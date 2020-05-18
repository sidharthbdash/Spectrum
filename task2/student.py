#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:15:08 2020

@author: jenar
"""
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset and creating dataframe
dataset=pd.read_csv("student-math.csv",sep=';')

#Adding 'total_grades' to the dataframe
dataset['final_grade'] = dataset['G1'] + dataset['G2'] + dataset['G3']

#Dropping 'G1','G2','G3'
dataset=dataset.drop(columns=['G1','G2','G3'])

#Replacing binary data values with '1' or '0'
dataset['school']=dataset['school'].map({"GP":'0',"MS":'1'})
dataset['sex']=dataset['sex'].map({"F":'0',"M":'1'})
dataset['address']=dataset['address'].map({"U":'0',"R":'1'})
dataset['famsize']=dataset['famsize'].map({"LE3":'0',"GT3":'1'})
dataset['Pstatus']=dataset['Pstatus'].map({"T":'0',"A":'1'})
dataset['schoolsup']=dataset['schoolsup'].map({"no":'0',"yes":'1'})
dataset['famsup']=dataset['famsup'].map({"no":'0',"yes":'1'})
dataset['paid']=dataset['paid'].map({"no":'0',"yes":'1'})
dataset['activities']=dataset['activities'].map({"no":'0',"yes":'1'})
dataset['nursery']=dataset['nursery'].map({"no":'0',"yes":'1'})
dataset['higher']=dataset['higher'].map({"no":'0',"yes":'1'})
dataset['internet']=dataset['internet'].map({"no":'0',"yes":'1'})
dataset['romantic']=dataset['romantic'].map({"no":'0',"yes":'1'})

#Boxplot 
dataset.boxplot(by='studytime',column=['final_grade'],grid=False)

#Scatter Plot between study time and final_grade
plt.scatter(dataset.studytime,dataset.final_grade,color='blue',s=25,marker=".")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")