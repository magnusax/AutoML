def library_config(names, nrow, ncol):
    library = {
        'logreg': _logreg_lib(nrow, ncol),
        'adaboost': _adaboost_lib(nrow, ncol),
        'nearestneigbors': _nneighbors_lib(nrow, ncol), 
        'sgd_log_loss': None, 
        'gaussian_nb': None, 
        'multinomial_nb': _nbayes_lib(nrow, ncol), 
        'bernoulli_nb': _nbayes_lib(nrow, ncol),
        'random_forest': _randforest_lib(nrow, ncol),
    }
    return [
        (name, grid) for name, lib in library.items() if (name in names) and (lib is not None)
    ]
       
def _logreg_lib(nrow, ncol):
    grid = {
        'category': 'continuous', 
        'method': 'sample', 
        'prior': 'loguniform', 
        'low': 1e-7, 
        'high': 1e+4, 
        'numval': 50
     }
    return [ 
        {'param': 'C', 
         'grid': grid, 
         'config': {'penalty': 'l1'}},
        
        {'param': 'C', 
         'grid': grid, 
         'config': {'penalty': 'l2'}}
    ]

def _adaboost_lib(nrow, ncol):
    values = [
        v for v in (1, 2, 4, 6, 8, 12, 16, 20) if v<=ncol
    ]
    return [
        {'param': 'base_estimator__max_features', 
         'grid': {'category': 'discrete', 'method': 'take', 'values': values},
         'config': {'n_estimators': 512, 'base_estimator__criterion': 'gini'},}
    ]

def _nneighbors_lib(nrow, ncol):
    values = [
        v for v in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31) if v<=nrow
    ]
    values.append(nrow)
    return [
        {'param': 'n_neighbors', 
         'grid': {'category': 'discrete', 'method': 'take', 'values': values},
         'config': {'weights': 'distance'}},
        
        {'param': 'n_neighbors', 
         'grid': {'category': 'discrete', 'method': 'take', 'values': values},
         'config': {'weights': 'uniform'}}
    ]

def _nbayes_lib(nrow, ncol):
    return [
        {'param': 'alpha', 
         'grid': {'category': 'continuous', 
                  'method': 'sample', 
                  'prior': 'uniform', 
                  'low': 0., 
                  'high': 1., 
                  'numval': 20},
         'config': {'fit_prior': True}}
    ]

def _randforest_lib(nrow, ncol):
        values = [
            v for v in (1, 2, 4, 6, 8, 12, 16, 20, 24) if v <= ncol
        ]
    return [
        {'param': 'max_features',
         'grid': {'category': 'discrete', 'metod': 'take', 'values': values},
         'config': {'n_estimators': 1024},}
    ]

def _svm_lib():
    pass

def _dtree_lib():
    pass

def _nn_lib():
    pass