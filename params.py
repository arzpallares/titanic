"""
References:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
"""
# Decision Tree
tre_params = [
        {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', None]
    }
]
# Random Forest
for_params = [
    {
        'n_estimators': [50, 100, 250, 500, 600, 750, 1000, 1500],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', None]
    }
]
# Linear Support Vector Classifier
lin_params = [
    {
        'penalty': ['l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'C': [0.01, 0.1, 1.0],
        'max_iter': [10000, int(1e5), int(1e6)]  
    },
    {
        'penalty': ['l1'],
        'loss': ['squared_hinge'],
        'dual': [True, False],
        'C': [0.01, 0.1, 1.0],
        'max_iter': [10000, int(1e5), int(1e6)]  
    },
]
# Support Vector Classifier
svc_params = [
    {
        'C': [0.01, 0.1, 1, 5, 10, 25, 30, 50],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'coef0': [0.0, 0.3, 0.5, 1.0, 0.5, 2.0],
        'degree': [2, 3],
        'max_iter': [int(1e6), int(1e7)]
    },
]
# Stochastic Gradient
sgd_params = [
    {
        'loss': ['hinge', 'perceptron', 'log_loss', 'squared_hinge'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.01, 0.1, 0.25, 0.50, 1],
        'l1_ratio': [0.30, 0.50, 0.75, 0.90],
        'max_iter': [100000, int(1e6), int(1e7)]
    },
]
# K-Nearest Neighbors
knn_params = [
    {
        'n_neighbors': [3, 4, 5, 6, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['minkowski'],
        'p': [1, 1.5, 2, 2.5, 3],
        'n_jobs': [None, -1]
    },
    {
        'n_neighbors': [3, 4, 5, 6, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'],
        'n_jobs': [None, -1]
    }
]
# Naive Bayes
bay_params = [
    {
        'var_smoothing': [1e-11, 1e-10, 1e-9]
    }
]