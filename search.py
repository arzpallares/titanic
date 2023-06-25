from sklearn.model_selection import GridSearchCV

def run_search(clf=None, params=None, cv=10, X=None, y=None):
    search = GridSearchCV(clf, params, cv=cv, scoring='accuracy', return_train_score=True)
    search.fit(X, y)
    return search.best_params_