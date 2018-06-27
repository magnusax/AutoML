import numpy as np


def library_config(names, nrow, ncol):
    """Provide information on how to generate ensemble."""
    library = {
        'logreg': logreg_lib(nrow, ncol),
        'adaboost': adaboost_lib(nrow, ncol),
        'knn': nearest_neighbors_lib(nrow, ncol), 
        'sgd_hinge': gradient_descent_lib(), 
        'gaussian_nb': gaussian_naive_bayes(), 
        'svm': svm_lib(),
        'multinomial_nb': naive_bayes_lib(nrow, ncol), 
        'bernoulli_nb': naive_bayes_lib(nrow, ncol),
        'random_forest': random_forest_lib(nrow, ncol),
        'tree': decision_tree_lib(nrow, ncol),
        'xgboost': xgboost_lib(nrow, ncol) }
    
    return list((name, grid) for name, grid in library.items() 
        if (name in names) and (grid is not None))


def get_config(hyper_param, fixed_params, grid):
    return [{'param': hyper_param, 
             'config': fixed_params, 'grid': grid}]


def get_grid(category, method, 
             values=None, prior=None, low=None, high=None, numval=None):
    """Generate a configuration grid used when building our ensembler.
    
    Input:
    ---------
        category : str 
            'discrete' or 'continuous'
            
        method : str
            'sample' or 'take'
            
        values : iterable, array-like
            Used only when method=='take'. 
            Specifies values of hyperparameter.
            
        prior : str
            Specifies how to sample hyperparameter 
            ('uniform' or 'loguniform'). 
            Used when method=='sample'.
            
        low : integer or float:
            Sampling lower bound.
            
        high : integer or float
            Sampling upper bound.
            
        numval : integer
            Number of values to sample.
            
    Returns:
    ---------    
    Dictionary with conditional keys.
    """
    if not category in ('discrete', 'continuous'):
        raise Exception("Invalid 'category' argument.")
    if not method in ('sample', 'take'):
        raise Exception("Invalid 'method' argument.")
    
    config = {}
    config['category']=category
    config['method']=method
    
    if method=='sample':
        if not prior in ('uniform', 'loguniform'):
            raise Exception("Invalid 'prior' argument.")
        config['prior']=prior
        
        if (low is None) or (high is None) or (numval is None):
            raise Exception("'low', 'high' and 'numval' must be set when sampling")
        config.update({'low':low, 'high':high, 'numval':int(numval)})
        
    elif method=='take':
        if len(values)==0:
            raise ValueError("'values' is empty")
        if not isinstance(values, list):
            values = list(values)
        config['values']=values
    return config


######################################
#         MODEL LIBRARIES
######################################


def xgboost_lib(nrow, ncol):
    
    depths = get_grid('discrete', 'take', values=list(range(1,21)))
    estimators = get_grid('discrete','take', values=[int(n) for n in np.linspace(10, 1000, 31, endpoint=True)])
    
    return get_config('max_depth', {'n_estimators':128}, depths)+\
           get_config('max_depth', {'n_estimators':512}, depths)+\
           get_config('n_estimators', {'max_depth': 2}, estimators) 

def logreg_lib(nrow, ncol):
    Cs = get_grid('continuous', 'sample', prior='loguniform', low=1e-8, high=1e+8, numval=100)
    
    return get_config('C', {'penalty': 'l1'}, Cs)+\
           get_config('C', {'penalty': 'l2'}, Cs)
    
def adaboost_lib(nrow, ncol):
    max_feats = [f for f in list(range(1,51)) if f<=ncol]
    grid = get_grid('discrete', 'take', values=max_feats)
    
    return get_config('base_estimator__max_features', 
        {'n_estimators': 128, 'base_estimator__criterion': 'gini'}, grid)+\
           get_config('base_estimator__max_features', 
        {'n_estimators': 256, 'base_estimator__criterion': 'entropy'}, grid)

def nearest_neighbors_lib(nrow, ncol):
    num_neighbors = [v for v in (int(2*n+1) for n in range(1,31)) if v<nrow]
    neighbors = get_grid('discrete', 'take', values=num_neighbors)
    
    return get_config('n_neighbors', {'weights': 'distance'}, neighbors)+\
           get_config('n_neighbors', {'weights': 'uniform'}, neighbors) 
    
def naive_bayes_lib(nrow, ncol):
    alphas = get_grid('continuous', 'sample', prior='uniform', low=0., high=1., numval=20)
    return get_config('alpha', {'fit_prior': True}, alphas)+\
           get_config('alpha', {'fit_prior': False}, alphas)
    
def gaussian_naive_bayes():
    return get_config('priors', {}, get_grid('discrete', 'take', values=[None]))

def random_forest_lib(nrow, ncol):
    max_feats = [f for f in list(range(1, 51)) if f<=ncol]
    maxfeats = get_grid('discrete', 'take', values=max_feats)
    
    return get_config('max_features', {'n_estimators': 512, 'criterion':'gini'}, maxfeats)+\
           get_config('max_features', {'n_estimators': 512, 'criterion':'entropy'}, maxfeats)

def svm_lib():
    alphas = get_grid('continuous', 'sample', prior='loguniform', low=1e-8, high=1e+8, numval=100)
    
    return get_config('model__alpha', {'model__penalty': 'l1'}, alphas)+\
           get_config('model__alpha', {'model__penalty': 'l2'}, alphas)

def gradient_descent_lib():
    alphas = get_grid('continuous', 'sample', prior='loguniform', low=1e-8, high=1e+8, numval=100)
    
    return get_config('alpha', {'penalty': 'l2', 'max_iter': 10}, alphas)+\
           get_config('alpha', {'penalty': 'l1', 'max_iter': 25}, alphas)
    
def decision_tree_lib(nrow, ncol):
    max_feats = get_grid('continuous', 'sample', prior='uniform', low=0.15, high=0.95, numval=100)
    depths = get_grid('discrete', 'take', values=list(range(1,21)))
    return get_config('max_features', {}, max_feats)+\
           get_config('max_depth', {}, depths) 
    
    

def neuralnet_lib():
    pass