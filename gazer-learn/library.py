def library_config(names, nrow, ncol):
    """ Provide information on how to generate ensemble """
    
    library = {
        'logreg': logreg_lib(nrow, ncol),
        'adaboost': adaboost_lib(nrow, ncol),
        'nearestneigbors': nearest_neighbors_lib(nrow, ncol), 
        'sgd_log_loss': None, 
        'gaussian_nb': None, 
        'multinomial_nb': naive_bayes_lib(nrow, ncol), 
        'bernoulli_nb': naive_bayes_lib(nrow, ncol),
        'random_forest': random_forest_lib(nrow, ncol),
        'xgboost': xgboost_lib(nrow, ncol),}
    return [
        (name, grid) for name, grid in library.items() 
        if (name in names) and (grid is not None)]

def get_config(hyper_param, fixed_params, grid):
    return list({
        'param':hyper_param, 
        'config':fixed_params, 
        'grid':grid})

def get_grid(category, method, values=None, prior=None, low=None, high=None, numval=None):
    """
    Generate a configuration grid used when building our ensembler.
    
    Input:
    ---------------
        category (str): 
            'discrete' or 'continuous'
        method (str):
            'sample' or 'take'
        values (iterable, array-like):
            Used only when method=='take'. Specifies values of hyperparameter.
        prior (str):
            Specifies how to sample hyperparameter ('uniform' or 'loguniform'). 
            Used when method=='sample'.
        low (integer, float):
            Sampling lower bound.
        high (integer, float):
            Sampling upper bound.
        numval (integer):
            Number of values to sample.
            
    Returns:
    ---------------    
        dict[category:str, method:str, values:list()]
        or
        dict[category:str, method:str, prior:str, low:(int,float), high:(int,float), numval:int)]
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
    
def xgboost_lib(nrow, ncol):
    depth_grid=get_grid(
        'discrete', 
        'take', 
        values=[1,2,3,4,5,6,7,8,9,10])    
    return get_config('max_depth', {'n_estimators':100}, depth_grid)

def logreg_lib(nrow, ncol):
    C_grid = get_grid(
        'continuous', 
        'sample', 
        prior='loguniform', low=1e-7, high=1e+4, numval=50)
    return get_config('C', {'penalty': 'l1'}, C_grid)+\
           get_config('C', {'penalty': 'l2'}, C_grid)

def adaboost_lib(nrow, ncol):
    max_feats = [
        f for f in (1, 2, 4, 6, 8, 12, 16, 20) if f<=ncol]
    grid = get_grid('discrete', 'take', values=max_feats)
    return get_config('base_estimator__max_features', 
                      {'n_estimators': 512, 
                       'base_estimator__criterion': 'gini'},
                      grid)

def nearest_neighbors_lib(nrow, ncol):
    num_neighbors = [
        v for v in (1, 3, 5, 7, 9, 11, 13, 
                    15, 17, 19, 21, 23, 25, 
                    27, 29, 31) if v<nrow]
    grid = get_grid('discrete', 'take', values=num_neighbors)  
    return get_config('n_neighbors', {'weights': 'distance'}, grid)+\
           get_config('n_neighbors', {'weights': 'uniform'}, grid) 

def naive_bayes_lib(nrow, ncol):
    grid = get_grid(
        'continuous', 
        'sample', 
        prior='uniform', low=0., high=1., numval=20)
    return get_config('alpha', {'fit_prior':True}, grid)

def random_forest_lib(nrow, ncol):
    max_feats = [
        f for f in (1, 2, 4, 6, 8, 12, 16, 20, 24) 
        if f<=ncol]
    grid = get_grid('discrete', 'take', values=max_feats)
    return get_config('max_features', {'n_estimators':1024}, grid)

def svm_lib():
    pass

def decision_tree_lib():
    pass

def neuralnet_lib():
    pass