import json
import numpy as np

grid_search = {}


#  ██████ ██       █████  ███████ ███████ ██ ███████ ██ ███████ ██████
# ██      ██      ██   ██ ██      ██      ██ ██      ██ ██      ██   ██
# ██      ██      ███████ ███████ ███████ ██ █████   ██ █████   ██████
# ██      ██      ██   ██      ██      ██ ██ ██      ██ ██      ██   ██
#  ██████ ███████ ██   ██ ███████ ███████ ██ ██      ██ ███████ ██   ██




grid_search['LogisticRegression'] = {
    'penalty': ('l1', 'l2', 'elasticnet'),
    'C': [10**x for x in range(-3,3)],
    'fit_intercept': [True, False],
    'max_iter': [int(x) for x in range(100, 10000, 1000)]
}

grid_search['DecisionTreeClassifier'] = {
    'criterion': ('gini', 'entropy'),
    'splitter': ('best', 'random'),
    'max_depth': [int(x) for x in np.linspace(1, 100, num = 5)],
    'min_samples_split': [x for x in np.linspace(0.1, 1.0, 10, endpoint=True)],
    'min_samples_leaf': [x for x in np.linspace(0.1, 0.5, 5, endpoint=True)],
    'max_features': ('sqrt', 'log2', 'None')
}

grid_search['SVC'] = {
    'C': [10**x for x in range(-3,3)],
    # 'kernel': ('linear', 'rbf', 'poly'),
    'gamma': [10**x for x in range(-3,3)],
}

grid_search['KNeighborsClassifier'] = {
    'n_neighbors': [int(x) for x in range(1,30)],
    'weights': ('uniform', 'distance'),
    'leaf_size': [int(x) for x in range(10, 100, 20)],
    'p': [int(x) for x in range(1,6)],
}

grid_search['MLPClassifier'] = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [10**x for x in range(-3,2)],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'max_iter': [100, 200, 600],
    'early_stopping': [True]
}

grid_search['RandomForestClassifier'] = {
    'n_estimators': [int(x) for x in np.linspace(10, 100, 5, endpoint=True)],
    'max_depth': [x for x in np.linspace(1, 32, 32, endpoint=True)],
    'max_features': [int(x) for x in range(1,4)],
}

grid_search['GradientBoostingClassifier'] = {
    'learning_rate': [x for x in np.linspace(0.01, 1.0, 6, endpoint=True)],
    'loss': ('deviance', 'exponential'),
    'n_estimators': [50, 100, 150],
    'max_depth': [x for x in np.linspace(1, 32, 32, endpoint=True)],
    'max_features': [int(x) for x in range(1,4)],
}


# ██████  ███████  ██████  ██████  ███████ ███████ ███████  ██████  ██████
# ██   ██ ██      ██       ██   ██ ██      ██      ██      ██    ██ ██   ██
# ██████  █████   ██   ███ ██████  █████   ███████ ███████ ██    ██ ██████
# ██   ██ ██      ██    ██ ██   ██ ██           ██      ██ ██    ██ ██   ██
# ██   ██ ███████  ██████  ██   ██ ███████ ███████ ███████  ██████  ██   ██



grid_search['LinearRegression'] = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

grid_search['DecisionTreeRegressor'] = {
    'max_depth': [int(x) for x in np.linspace(1, 100, 5)],
    'min_samples_split': [0.001*x for x in range(100)],
    'min_samples_leaf': [0.001*x for x in range(100)],
}

grid_search['RandomForestRegressor'] = {
    'n_estimators': [int(x) for x in np.linspace(100, 600, 50, endpoint=True)],
    'max_depth': [int(x) for x in np.linspace(1, 50, 30, endpoint=True)],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf':  [x for x in np.linspace(0.1, 1, 10, endpoint=True)],
    'min_samples_split': [x for x in np.linspace(0.1, 1, 10, endpoint=True)],
    'bootstrap': [True, False]
}

grid_search['BaggingRegressor'] = {
    'n_estimators':  [int(x) for x in np.linspace(100, 800, 50, endpoint=True)],
    'max_samples': [x for x in np.linspace(0.8, 1.0, 3, endpoint=True)],
    'max_features': [x for x in np.linspace(0.6, 0.8, 3, endpoint=True)],
    'bootstrap': [True, False]
}

grid_search['AdaBoostRegressor'] = {
    'n_estimators':  [int(x) for x in np.linspace(50, 300, 50, endpoint=True)],
    'loss' : ['linear', 'square', 'exponential'],
    'learning_rate': [10**x for x in range(-3,2)],
}

grid_search['GradientBoostingRegressor'] = {
    # 'learning_rate': [x for x in np.linspace(0.01, 1.0, 6, endpoint=True)],
    'learning_rate': [10**x for x in range(-3,2)],
    'loss': ('ls', 'lad'),
    'n_estimators': [int(x) for x in np.linspace(100, 600, 50, endpoint=True)],
    'max_depth': [x for x in np.linspace(5, 100, 5, endpoint=True)],
    'max_features': ['auto', 'sqrt', 'log2'],
}

grid_search['XGBRegressor'] = {
    'n_estimators':  [int(x) for x in np.linspace(100, 600, 50, endpoint=True)],
    'max_depth': [x for x in np.linspace(5, 100, 5, endpoint=True)],
    'learning_rate': [x for x in np.linspace(0.3, 0.5, 3, endpoint=True)],
    'booster': ['gbtree', 'gblinear'],
}

with open('param_search', 'w') as f:
    json.dump(grid_search, f)
