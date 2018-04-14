#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:37:04 2018

@author: Nord
"""

"""
Aryana Collins Jackson
R00169199
Assignment 1: Model 2
"""
# Import necessary libraries and set styles

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_color_codes("pastel")

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import SVM
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import metrics

from sklearn import svm

from sklearn import model_selection
from sklearn import metrics
from sklearn import datasets
from sklearn import tree

from sklearn import preprocessing

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
    data = pd.read_csv('student/student-mat.csv', sep=';')
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
 
#    Remove the grade values to determine what other variables influence G3. 

    X = data.drop(['G3', 'G2', 'G1'], axis=1)
    
    X = pd.get_dummies(X)
    
    for name, clf in zip(names, clf_list):
        print(name, end=': ')
        print(cross_val_score(clf, X, y, cv=5).mean())
        
#    I am running feature importances again to get a better idea of what's 
#    going on here and why these results are so low

    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], 
              importances[indices[f]]))
        
#    Here I am running results of G1->G3, G2->G3, and absenses->G3 prediction
#    for 5-fold cross-validation. I am simply changing this bit of code for
#    length purposes. The other two variables are inputting in the second line
#    in "X = data["variable"]
    
    y = data['G3']
    X = data['absences']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred = linreg.predict(X)
    fig = plt.figure()
    plt.scatter(X,y, color='black', alpha=.1)
    fig.suptitle('Relationship Absences - Grades G3', fontsize=12)
    plt.xlabel('Absences', fontsize=12)
    plt.ylabel('Grades G3', fontsize=12)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot(X,y_pred, color='red')

    print("Mean squared error: %.4f"
          % np.mean((linreg.predict(X) - y) ** 2))

    print('Variance score: %.4f' % linreg.score(X, y))
    
    print('Cross-val score: %.4f' % cross_val_score(clf, X, y, cv=5).mean())
    
#    Here I am running results of G1->G3, G2->G3, and absences->G3 prediction
#    for 10-fold cross-validation. I am simply changing this bit of code for
#    length purposes. The other two variables are inputting in the second line
#    in "X = data["variable"]
    
    y = data['G3']
    X = data['G1']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred = linreg.predict(X)
    fig = plt.figure()
    plt.scatter(X,y, color='black', alpha=.1)
    fig.suptitle('Relationship Grades G1 - Grades G3', fontsize=12)
    plt.xlabel('Grades G1', fontsize=12)
    plt.ylabel('Grades G3', fontsize=12)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot(X,y_pred, color='red')

    print("Mean squared error: %.4f"
          % np.mean((linreg.predict(X) - y) ** 2))

    print('Variance score: %.4f' % linreg.score(X, y))
    
    print('Cross-val score: %.4f' % cross_val_score(linreg, X, y, cv=10).mean())

############################################################    CLASSIFICATIONS
    
    #    ------------------------------ INDIVIDUAL 

#    Test classification models: logistic regression, decision trees, random 
#    forests, and support vector classification. The following code prints out
#    the results of the four classification models on the data. 
    
    data = pd.read_csv('student/student-mat.csv', sep=';')

    X = data.drop('G3', axis=1)
    y = data['G3']

    # Encoding our categorical columns in X
    labelEncoder = preprocessing.LabelEncoder()
    cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        X[col] = labelEncoder.fit_transform(X[col])

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=52)
    # Logistic Regression as baseline, then exploring tree-based methods

    keys = []
    scores = []
    
    from sklearn import svm

    models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=300, random_state=52), 'SVC': svm.SVC()}

    for k,v in models.items():
        mod = v
        mod.fit(X_train, y_train)
        pred = mod.predict(X_test)
        print('Results for: ' + str(k) + '\n')
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        acc = accuracy_score(y_test, pred)
        print("accuracy is "+ str(acc)) 
        print('\n' + '\n')
        keys.append(k)
        scores.append(acc)
        table = pd.DataFrame({'model':keys, 'accuracy score':scores})

    print(table)
    
#    We want to now compare cross-validation to nested cross validation. This
#    is done with SVC. Code is taken from scikit documentation, cited in paper

    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

    X = data.drop('G3', axis=1)
    y = data['G3']

    # Encoding our categorical columns in X
    labelEncoder = preprocessing.LabelEncoder()
    cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        X[col] = labelEncoder.fit_transform(X[col])
    
    from sklearn import svm

    svm = svm.SVC(kernel="rbf")
    
    NUM_TRIALS = 5
    
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)
    
    p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
    
    for i in range(NUM_TRIALS):

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()
    
    score_difference = non_nested_scores - nested_scores

    print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))
    print("Cross val scores mean: ", score_difference.mean())
    print("Nested scores mean: ", nested_scores[i].mean())
    print("Non-nested scores mean: ", non_nested_scores[i].mean())
    
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation",
          x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    plt.show()   
#    ------------------------------ PASS/FAILS
#    We want to do the same thing, but for pass/fails, not for each individual 
#    grade. So y is turned into a boolean array.

    data = pd.read_csv('student/student-mat.csv', sep=';')
    
    X = data.drop('G3', axis=1)
    y = data['G3'] 
    
    y = y >= 10

    # Encoding our categorical columns in X
    labelEncoder = preprocessing.LabelEncoder()
    cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        X[col] = labelEncoder.fit_transform(X[col])
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                        random_state=52)
    
    from sklearn import svm
    
    keys = []
    scores = []
    models = {'Logistic Regression': LogisticRegression(), 
              'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=300, 
                                                  random_state=52), 
                                                  'SVC': svm.SVC()}

    for k,v in models.items():
        mod = v
        mod.fit(X_train, y_train)
        pred = mod.predict(X_test)
        print('Results for: ' + str(k) + '\n')
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        acc = accuracy_score(y_test, pred)
        print("accuracy is "+ str(acc)) 
        print('\n' + '\n')
        keys.append(k)
        scores.append(acc)
        table = pd.DataFrame({'model':keys, 'accuracy score':scores})

    print(table)
    
#    We want to improve the accuracy of the decision tree and SVC only since
#    those had the highest accuracy. First we try nested cross-validation. 
    
    data = pd.read_csv('student/student-mat.csv', sep=';')
    
    X = data.drop('G3', axis=1)
    y = data['G3'] 
    
    y = y >= 10

    # Encoding our categorical columns in X
    labelEncoder = preprocessing.LabelEncoder()
    cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        X[col] = labelEncoder.fit_transform(X[col])
        
    from sklearn import svm

    svm = svm.SVC(kernel="rbf")
    
    NUM_TRIALS = 5
    
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)
    
    p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
    
    for i in range(NUM_TRIALS):

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()
        
    score_difference = non_nested_scores - nested_scores

    print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))
    print("Cross val scores mean: ", score_difference.mean())
    print("Nested scores mean: ", nested_scores[i].mean())
    print("Non-nested scores mean: ", non_nested_scores[i].mean())
    
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation",
          x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    plt.show()    
#    ------------------------------ THREE-LEVEL

#    We want to do the same thing, but for high passes, low passes, and fails.

    data = pd.read_csv('student/student-mat.csv', sep=';')
    
    X = data.drop('G3', axis=1)
    y = data['G3'] 
    
    grades = []

    # For each row in the column,
    for row in data['G3']:
        # if more than a value,
        if row > 17:
            # Append a letter grade
            grades.append('A')
        elif row > 10:
            grades.append('C')
        else:
            grades.append('F')
    
    y = grades

    # Encoding our categorical columns in X
    labelEncoder = preprocessing.LabelEncoder()
    cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        X[col] = labelEncoder.fit_transform(X[col])
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                        random_state=52)

    from sklearn import svm

    keys = []
    scores = []
    models = {'Logistic Regression': LogisticRegression(), 
              'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=300, 
                                                  random_state=52), 
                                                  'SVC': svm.SVC()}

    for k,v in models.items():
        mod = v
        mod.fit(X_train, y_train)
        pred = mod.predict(X_test)
        print('Results for: ' + str(k) + '\n')
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        acc = accuracy_score(y_test, pred)
        print("accuracy is "+ str(acc)) 
        print('\n' + '\n')
        keys.append(k)
        scores.append(acc)
        table = pd.DataFrame({'model':keys, 'accuracy score':scores})

    print(table)
    
#    Same process - nested cross-val
    
    data = pd.read_csv('student/student-mat.csv', sep=';')
    
    X = data.drop('G3', axis=1)
    y = data['G3'] 
    
    grades = []

    # For each row in the column,
    for row in data['G3']:
        # if more than a value,
        if row > 17:
            # Append a letter grade
            grades.append('A')
        elif row > 10:
            grades.append('C')
        else:
            grades.append('F')
    
    y = grades

    # Encoding our categorical columns in X
    labelEncoder = preprocessing.LabelEncoder()
    cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        X[col] = labelEncoder.fit_transform(X[col])
        
    from sklearn import svm

    svm = svm.SVC(kernel="rbf")
    
    NUM_TRIALS = 5
    
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)
    
    p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
    
    for i in range(NUM_TRIALS):

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()
    
    score_difference = non_nested_scores - nested_scores

    print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))
    print("Cross val scores mean: ", score_difference.mean())
    print("Nested scores mean: ", nested_scores[i].mean())
    print("Non-nested scores mean: ", non_nested_scores[i].mean())
    
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation",
          x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    plt.show()    
#
# Visualisations function is not included in main function to avoid clutter
#visualisations()

main()

