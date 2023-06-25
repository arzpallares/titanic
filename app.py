"""
Titanic Survival Prediction Project

Based on the Titanic Survival Challenge in Kaggle

Objetive:
- Predict whenever or not a given passanger would survive the Titanic's sinking
- Determine the most important features
"""
# Core imports
import numpy as np
import pandas as pd

# Sub modules import
import cleanup
import search
from params import *

# Load the datasets into Dataframes
train_data = pd.read_csv('data/train-titanic.csv', index_col='PassengerId')
test_data = pd.read_csv('data/test-titanic.csv', index_col='PassengerId')

X = cleanup.fix_columns(train_data)

cleaned_test = cleanup.fix_columns(test_data)

cleanup.fix_dtypes(X)

y = X.pop('Survived')

"""
Feature scaling can sometimes help to improve the performance of the models
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
resX = scaler.fit_transform(X)

"""
Split the sample into train and testing
"""
X_train, X_test, y_train, y_test = train_test_split(resX, y, test_size=0.30, random_state=42)

"""
Model Selection:
Three criteria will be used to select the best models to fulfill the task:
1- Prediction Type
2- Dataset Shape
3- Model Performance
----------------------------------------------------------
Multivariative, Binary Classification
m>n
acceptable performance must be at least 90% accuracy
"""

# Model imports
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

"""
Search for the most optimal parameter configurations for the model
"""


tre_params = search.run_search(
        DecisionTreeClassifier(), tre_params, 10, X_train, y_train)
bay_params = search.run_search(
        GaussianNB(), bay_params, 10, X_train, y_train)
for_params = search.run_search(
        RandomForestClassifier(), for_params, 10, X_train, y_train)
knn_params = search.run_search(
        KNeighborsClassifier(), knn_params, 10, X_train, y_train)
lin_params = search.run_search(
        LinearSVC(), lin_params, 10, X_train, y_train)
svc_params = search.run_search(
        SVC(), svc_params, 10, X_train, y_train)
sgd_params = search.run_search(
         SGDClassifier(), sgd_params, 10, X_train, y_train)

tre_clf = DecisionTreeClassifier(**tre_params)
bay_clf = GaussianNB(**bay_params)
for_clf = RandomForestClassifier(**for_params)
knn_clf = KNeighborsClassifier(**knn_params)
lin_clf = LinearSVC(**lin_params)
sgd_clf = SGDClassifier(**sgd_params)
svc_clf = SVC(**svc_params)


vot_clf = VotingClassifier(
    estimators=[
                ('knn', knn_clf),
                ('sgd', sgd_clf),
                ('svc', svc_clf),
                ('tre', tre_clf),
        ],
    voting='soft',
)

from sklearn.metrics import accuracy_score

for clf in (tre_clf, for_clf, bay_clf, knn_clf, svc_clf, sgd_clf, lin_clf, vot_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))