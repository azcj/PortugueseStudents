#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:04:55 2018

@author: Nord
"""

"""
Aryana Collins Jackson
R00169199
Assignment 1
"""

# This code is largely unedited from the original version. The only difference 
# is the addition of the main function and therefore one extra parameter in the 
# regression_explore function and the addition of the visualisations function.

# Import necessary libraries and set styles

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_color_codes("pastel")

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

def visualisations(data):
    
    #    Series of visualisations - interpretations in report
#    Visualisation by sex
    
    f, ax = plt.subplots(figsize=(4, 4))
    plt.pie(data['sex'].value_counts().tolist(), 
        labels=['Female', 'Male'], colors=['#ffd1df', '#a2cffe'], 
        autopct='%1.1f%%', startangle=90)
    axis = plt.axis('equal')
    
#    Visualisation by age
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.distplot(data['age'],  
             hist_kws={"alpha": 1, "color": "#a2cffe"}, 
             kde=False, bins=8)
    ax = ax.set(ylabel="Count", xlabel="Age")
    
#    Visualisation by study time
    
    f, ax = plt.subplots(figsize=(4, 4))
    plt.pie(data['studytime'].value_counts().tolist(), 
        labels=['2 to 5 hours', '<2 hours', '5 to 10 hours', '>10 hours'], 
        autopct='%1.1f%%', startangle=0)
    axis = plt.axis('equal')
    
#    Visualisation by romantic status
    
    f, ax = plt.subplots(figsize=(4, 4))
    plt.pie(data['romantic'].value_counts().tolist(), 
        labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
    axis = plt.axis('equal')
    
#    Visualisations:
#    Workday alcohol consumption: number from 1 (very low) to 5 (very high)
#    Weekend alcohol consumption: number from 1 (very low) to 5 (very high)
#    Health - current health status: number from 1 (very bad) to 5 (very good)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.distplot(data['Walc'],  
             hist_kws={"alpha": 1, "color": "#a2cffe"}, 
             kde=False, bins=4)
    ax = ax.set(ylabel="Students", xlabel="Weekend Alcohol Consumption")
    
    plot1 = sns.factorplot(x="Walc", y="health", hue="sex", data=data)
    plot1.set(ylabel="Health", xlabel="Weekend Alcohol Consumption")

    plot2 = sns.factorplot(x="Dalc", y="health", hue="sex", data=data)
    plot2.set(ylabel="Health", xlabel="Workday Alcohol Consumption")
    
#    Visualisation: Final grade (number from 0 to 20)

    plot1 = sns.factorplot(x="G3", y="Walc", data=data)
    plot1.set(ylabel="Final Grade", xlabel="Weekend Alcohol Consumption")

    plot2 = sns.factorplot(x="G3", y="Dalc", data=data)
    plot2.set(ylabel="Final Grade", xlabel="Workday Alcohol Consumption")

def main():
    data = pd.read_csv('student/student-por.csv')
    data.head(5)
    
#    Begin regression model by setting x and y
    
    y = data['G3']
    X = data.drop(['G3'], axis=1)
    X = pd.get_dummies(X)
    
#    We set up the regression modelling with four different models. The for
#    loop iterates through the four models and through the dataset with the G3
#    variable dropped. The returned result (printed) is the mean of five-fold 
#    cross-validation for each model.
    
    names = ['DecisionTreeRegressor', 'LinearRegression', 'Ridge', 'Lasso']

    clf_list = [DecisionTreeRegressor(),
            LinearRegression(),
            Ridge(),
            Lasso()]
    
    for name, clf in zip(names, clf_list):
        print(name, end=': ')
        print(cross_val_score(clf, X, y, cv=5).mean())
        
#    Feature importances are completed next. First, the decision tree model is 
#    fit.
    
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], 
              importances[indices[f]]))
    
#    Results: G2 accounts for 81.74% of the G3 values. We can remove this 
#    value to determine what other variables influence G3. 

    X = data.drop(['G3', 'G2', 'G1'], axis=1)
    
    X = pd.get_dummies(X)
    
    for name, clf in zip(names, clf_list):
        print(name, end=': ')
        print(cross_val_score(clf, X, y, cv=5).mean())
        
#    The results are extremely low, resulting in models that simply don't work. 
#    Not in the original code - I am running feature importances again to get 
#    a better idea of what's going on here and why these results are so low

    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], 
              importances[indices[f]]))
    
# Visualisations function is not included in main function to avoid clutter
#visualisations()

main()
