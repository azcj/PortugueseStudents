#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:16:55 2018

@author: Nord
"""

"""
Aryana Collins Jackson
R00169199
Assignment 1
"""

# import libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import scipy
import matplotlib.pyplot as plt

#    We move on to regression modelling. We drop the variables 'G1.y', 'G2.y' 
#    and 'G3.y', because they're the target variables. The analysis from 
#    Dmitriy Batogov is used as a starting point. The analysis of the cross
#    val scores has been parameterised. 

def regression_explore(data,y,drop):
    Y = data[y]
    X = data.drop(drop, axis=1)
    X = pd.get_dummies(X)

    names = ['DecisionTreeRegressor', 'LinearRegression', 'Ridge', 'Lasso']

    clf_list = [DecisionTreeRegressor(),
            LinearRegression(),
            Ridge(),
            Lasso()]
            
    print('Models performance in: ' + str(y))
    print('------------------------')
    for name, clf in zip(names, clf_list):
        print(name, end=': ')
        print(cross_val_score(clf, X, Y, cv=5).mean())
        
#        Results for list a:
#            
#            Models performance in: G3.y
#            ------------------------
#            DecisionTreeRegressor: -0.428513294527
#            LinearRegression: 0.128844669426
#            Ridge: 0.203959055871
#            Lasso: 0.250043161244
#
#
#            Models performance in: G2.y
#            ------------------------
#            DecisionTreeRegressor: -0.273076913333
#            LinearRegression: 0.244250517022
#            Ridge: 0.346274708347
#            Lasso: 0.316640673154
#
#
#            Models performance in: G1.y
#            ------------------------
#            DecisionTreeRegressor: -0.455285053997
#            LinearRegression: -0.0741731848719
#            Ridge: 0.204108565568
#            Lasso: 0.304716660409

#        The variables G1, G2, and G3 do not show big relationships with the 
#        rest of the variables that we have. This could be due to the small 
#        sample size, or maybe we would need to collet data from some other 
#        variables. Either way, we'll have to change our approximation to this 
#        dataset if we want to extract some model.
#
#        In the correlation analysis in the main function, we noticed that the 
#        grades (G1, G2 and G3) have strong correlations between them. This 
#        indicates that someway the students who have big grades on the first 
#        period (G1) use to have big grades on the second period (G2) too. 
#        We'll check that the grades from the previous period can be a useful 
#        predictor from the grades on the next period:

def main():
    
    # import csv file with combined Portuguese and maths
    data = pd.read_csv('student/students.csv', sep=",")
       
#    Test if the distribution of the grades in the dataset follows a normal 
#    distribution by creating a dictionary with the grade columns. The for 
#    loops iterate through the dictionary and through each instance in the 
#    dataset. The stats are returned which show the result of the normal test
    
    grades=["G1.y","G2.y","G3.y"]
    norm=[]
    for i in grades:
        norm.append(scipy.stats.normaltest(data[i]))
    for i in range(0,len(grades)):
        print(grades[i])
        print(norm[i])
        print('----------')
    
#    The findings show that instances with a G2 grade (G1 found in the code in 
#    which Pandas are used to merge the CSV files) are the closest to a 
#    normal distribution. We test this with histograms of each. The for loop 
#    iterates through the dictionary containing the grades again and through
#    the dataset
    
    i = 1
    for w in grades:
        plt.subplot(3, 1, i)
        plt.tight_layout()
        i += 1
        plt.hist(data[w])
        plt.title(w)

#    There are more outliers in the plots for G2 and G3. Next we check for 
#    correlations between the pairs (G1 and G2, G2 and G3, and G1 and G3)
    
    corr = data.corr()
    for a in corr.columns:
        for b in corr.index:
            if (a != b) and (abs(corr[a][b]) >= 0.75):
                print(a,b,'-->',corr[a][b])
    
#    The results we're looking for are:
#        
#        G1.y G2.y --> 0.887906145682
#        G1.y G3.y --> 0.819048197226
#        G2.y G1.y --> 0.887906145682
#        G2.y G3.y --> 0.892443645076
#        G3.y G1.y --> 0.819048197226
#        G3.y G2.y --> 0.892443645076
#        
#    These are slightly different to the results in the original code. All 
#    variables show correlations above 0.75, which means it may be difficult
#    to do a PCA.    
    
#    Call the regression modelling function here 
    a = ['G3.y','G2.y','G1.y']
    for i in a:
        regression_explore(data, i, a)
        print('\n')
        
#    This time, we are experimenting with removing more variables just to see
#    what happens
    
    data2 = data.drop(['goout.x','goout.y','famrel.y','famrel.x',
          'freetime.x','freetime.y','studytime.x','studytime.y'], axis=1)

    a2 = ['G3.y','G2.y','G1.y']
    for i in a2:
        regression_explore(data2, i, a2)
        print('\n')
        
#    Examining the regression model just for G2 and G3   
    variables_explore = ['G2.y','G3.y']
    for b in variables_explore:
        regression_explore(data,b,b)
        print('\n')
        
#    This confirms the theory that the grades of the previous period are a good 
#    indicator of how the grades of the next period will be. Almost all of the 
#    models show a nice performance on the estimation, but I will choose the 
#    simple linear regression to perform this model:
    
    y = data['G2.y']
    X = data['G1.y']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred = linreg.predict(X)
    fig = plt.figure()
    plt.scatter(X,y, color='black', alpha=.1)
    fig.suptitle('Relatinship Grades G1 - Grades G2', fontsize=12)
    plt.xlabel('Grades G1', fontsize=12)
    plt.ylabel('Grades G2', fontsize=12)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot(X,y_pred, color='red')
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((linreg.predict(X) - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % linreg.score(X, y))
    
#    The darker of the dot, the more frequent that results are. The red line is 
#    our prediction model. We can see that the darker dots are located closer 
#    to the red line.
#
#    Original code: Mean squared error: 3.63, Variance score: 0.71
#    My code: Mean squared error: 1.29, Variance score: 0.79
#
#    Lets see if this regression works for the prediction of the 3rd period 
#    too:
    
    y = data['G3.y']
    X = data['G2.y']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred = linreg.predict(X)
    fig = plt.figure()
    plt.scatter(X,y, color='black', alpha=.1)
    fig.suptitle('Relatinship Grades G2 - Grades G3', fontsize=12)
    plt.xlabel('Grades G2', fontsize=12)
    plt.ylabel('Grades G3', fontsize=12)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot(X,y_pred, color='red')
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((linreg.predict(X) - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % linreg.score(X, y))
    
#    Original code: Mean squared error: 2.79, Variance score: 0.83
#    My code: Mean squared error: 1.76, Variance score: 0.80

#    It works even better for the 3rd period grades. This is because the 
#    correlation between the 2nd and the 3rd grades (~ 90%) is stronger than 
#    the correlation between the 1st and the 2nd period grades (~ 84%).

main()

# Conclusions
#
# We have not been able to find any significant correlations between the 
# students' alcohol habits and their grades on the math and portuguese courses. 
# This could be caused because we need to collect data from a bigger sample 
# size, or maybe because the grades are influenced by another different 
# variables that we're not considering on this database. Anyway, we've also 
# good news: it seems to be some kind of similar pattern between the grades 
# that the students get on the three periods, showing a continuation on the 
# school results. It will be needed some extra research in this field in order 
# to get more clear results.
