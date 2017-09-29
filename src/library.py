import sys

# Linear models
def _logreg_lib(nrow, ncol):
    grid = {'category':'continuous', 'method':'sample', 'prior':'loguniform', 'low':1e-7, 'high':1e+4, 'numval':50}
    return ( 
        {'param':'C', 
         'grid':grid, 
         'premise':{'penalty':'l1'}},
        {'param':'C', 
         'grid':grid, 
         'premise':{'penalty':'l2'}}
           )

# Ensembles
def _adaboost_lib(nrow, ncol):
    values = [v for v in (1, 2, 4, 6, 8, 12, 16, 20) if v<=ncol]
    return (
        {'param':'base_estimator__max_features', 
         'grid':{'category':'discrete', 'method':'take', 'values':values},
         'premise':{'n_estimators':512, 'base_estimator__criterion':'gini'}}
           )

# Memory based learning
def _nneighbors_lib(nrow, ncol):
    values = [v for v in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31) if v<=nrow]
    values.append(nrow)
    return (
        {'param': 'n_neighbors', 
         'grid':{'category': 'discrete', 'method':'take', 'values':values},
         'premise': {'weights': 'distance'}},
        {'param': 'n_neighbors', 
         'grid':{'category': 'discrete', 'method':'take', 'values':values},
         'premise': {'weights': 'uniform'}}
           )
        




# Support vector machines
def _svm_lib():
    pass

# Tree models
def _randforest_lib():
    pass

# Decision tree
def _dtree_lib():
    pass

# Network
def _nn_lib():
    pass
   
# Generative
def _nbayes_lib():
    pass



if __name__ == '__main__':
    sys.exit(-1)