"""
Credit:
--------
Methods iter_safe, and get_dicts are from:
    https://codereview.stackexchange.com/questions/128088/
        convert-a-dictionary-of-iterables-into-an-iterator-of-dictionaries
"""

import itertools
import pandas as pd
from tqdm import (tqdm, 
                  tqdm_notebook)


def iter_safe(value):
    if isinstance(value, str):
        value = (value,)
    try:
        iter(value)
    except TypeError:
        return (value,)
    else:
        return value
    
    
def get_dicts(d):
    keys, values_list = zip(*d.items())
    for values in itertools.product(*map(iter_safe, values_list)):
        yield dict(zip(keys, values))
        
        
def grid_search(learner, params, data, name='neuralnet', in_notebook='True'):
    """
    Grid search method customized for use with the Neural Network
    meta estimator, but works for any meta estimator in this library.
    
    - For native scikit-learn algorithms, it is recommended to use GridSearchCV
    class instead. This method does e.g. not implement parallel fitting. 
        
    Parameters:
    ------------
        learner : instance of GazerMetaLearner
            Grid search uses several instance methods in order to
            update parameters properly.
        
        name : str, 
            - A meta estimator identifier.
            - Must be present in learner.names.
        
        params : dict
            - A dictionary of parameters and corresponding values/iterables.
        
        data : dict
            - A dictionary containing train and CV data in the following format:
            data = {'train': (X_train, y_train), 'val': (X_val, y_val)}
            
        in_notebook, bool, default: True
            - Are we calling from a notebook or not. Has impact on choice
            of tqdm import.
            
    Returns:
    ---------
    Tuple of (best) parameter configuration, and pandas dataframe with complete
    overview of parameters and their train+validation scores for easy comparison.
    
    """    
    # Keep silent during search
    old_verbose = learner.verbose
    if old_verbose>0:
        learner.verbose = 0

    # Unpack data
    X_train, y_train = data['train']
    X_val, y_val = data['val']    
       
        
    # Internal convenience funcction
    def train_eval(pars):
        learner.update(name, pars)
        learner.fit(X_train, y_train)
        val = learner.evaluate(X_val, y_val)[name]
        train = learner.evaluate(X_train, y_train)[name]
        return {
            'train_loss':  train['loss'], 
            'train_score': train['score'],
            'val_loss':  val['loss'],                
            'val_score': val['score'] }
    
    params_scores = [] 
    number_of_fits = len(list(get_dicts(params)))    
    for param in tqdm_notebook(get_dicts(params), desc=name, total=number_of_fits):
        param.update(train_eval(param))
        params_scores.append(param) 
    
    # Restore when finished
    learner.verbose = old_verbose
    
    df = (pd.DataFrame
          .from_dict(params_scores)
          .sort_values(['val_score', 'val_loss'], ascending=[False, True])
          .reset_index())  
    
    config = df.T.to_dict()[0]
    remove = ('val_loss', 'val_score', 
              'train_loss', 'train_score')
    for key in remove: del config[key]
          
    return config, df        
        
