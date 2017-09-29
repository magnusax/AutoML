import sys

# Linear models
def _logreg_lib(nrow, ncol):
    return ( 
        {'param': 'C', 'category': 'continuous', 'low': 1e-7, 'high': 1e+4, 'numval': 50, 
         'premise': {'penalty': 'l1'}},
        {'param': 'C', 'category': 'continuous', 'low': 1e-7, 'high': 1e+4, 'numval': 50, 
         'premise': {'penalty': 'l2'}} 
           )

# Ensembles
def _adaboost_lib(nrow, ncol):
    values = [v for v in (1, 2, 4, 6, 8, 12, 16, 20) if v<=ncol]
    return (
        {'param': 'base_estimator__max_features', 'category': 'discrete', 
         'low': None, 'high': None, 'numval': None, 'values': values,
         'premise': {'n_estimators': 512, 'base_estimator__criterion': 'gini'}}        
           )

# Memory based learning
def _nneighbors_lib(nrow, ncol):
    values = [v for v in (1, 3, 5, 11, 13, 15, 21) if v<=nrow]
    values.append(nrow)
    return (
        {'param': 'n_neighbors', 'category': 'discrete', 
         'low': None, 'high': None, 'numval': None, 'values': values, 'premise': {'weights': 'distance'}},
        {'param': 'n_neighbors', 'category': 'discrete', 
         'low': None, 'high': None, 'numval': None, 'values': values, 'premise': {'weights': 'uniform'}}        
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