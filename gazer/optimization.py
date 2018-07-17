"""
Credit:
--------
Methods iter_safe, get_dicts are from:
    https://codereview.stackexchange.com/questions/128088/
    convert-a-dictionary-of-iterables-into-an-iterator-of-dictionaries

"""
import os
import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.externals import joblib
from sklearn.model_selection import ParameterSampler


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
        
        
# Internal convenience function
def _train_eval(name, learner, data, params):
    """ Update, train and evaluate. 
    """     
    # Update learner object with new params
    learner.update(name, params)
    
    # Train model, possibly save
    train_data, val_data = (data['train'], data['val'])
    learner.fit(*train_data)    
        
    # Evaluate on train + validation
    eval_kwargs = {'get_loss': True, 'verbose': 0}
    train = learner.evaluate(*train_data, **eval_kwargs)[name]
    
    if val_data is None:
        val = {'loss': np.nan, 'score': np.nan}
    else:
        val = learner.evaluate(*val_data, **eval_kwargs)[name]
        
    return {'train_loss': train['loss'], 'train_score': train['score'],
            'val_loss': val['loss'], 'val_score': val['score']}

        
        
def _save_model(estimator, file):
    
    dir_ = os.path.dirname(file)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    
    try:
        if hasattr(estimator, 'save'):   
            # If we are here then 'estimator' is most likely a keras model
            estimator.save(file, overwrite=True)       
        else:
            # Scikit-learn (default: overwrite)
            joblib.dump(estimator, file)
    except:
        _, desc, _ = sys.exc_info()
        raise Exception(
            "Could not save model: {}".format(desc))
    return

                 
def param_search(learner, param_grid, data, type_of_search, 
                 n_iter=12, name='neuralnet', modelfiles=[], top_n=1):
    """
    Grid search method customized for use with the Neural Network
    meta estimator, but works for any meta estimator in this library.
    
    - For native scikit-learn algorithms it is recommended to use GridSearchCV
    or RandomizedSearchCV classes instead. This method does e.g. not implement 
    distributed fitting. 
    
    
    Parameters:
    ------------
        learner : instance of GazerMetaLearner
            - Grid search uses several learner instance methods in order to
            update parameters properly.
        
        param_grid : dict
            - A dictionary of parameters and corresponding values/iterables.
        
        data : dict
            - A dictionary containing train and validation data in the following format:
            data = {'train': (X_train, y_train), 'val': (X_val, y_val)}
        
        type_of_search : str, values: ('random', 'grid')
            String that determines type of search mode. Possible values:
            - grid: perform a grid search.
            - random: perform a random search using n_iter draws from param_grid.
        
        n_iter : integer, default: 12
            Number of times we sample from param_grid when type_of_search='random'.
            - Not used when type_of_search='grid'.
            
        name : str, 
            - A meta estimator identifier.
            - Must be present in learner.names.
            
        modelfiles : list, default: empty list
            - List with N elements, where N are names of the top N models
            that should be saved to disk.
            - If empty, no files are saved.
            
        top_n : integer, default: 1 
            Number of models to return in the model config dict.
            
    Returns:
    ---------
    Tuple of (best) parameter configuration(s), and pandas dataframe with complete
    overview of parameters and their train + validation scores for easy comparison.
    
    - Number of parameter configurations set by 'top_n' parameter.
    
    - Note that the "best" configuration not necessarily corresponds to lowest 
    generalization error.
    
    """    
    # Keep silent during search
    old_verbose = learner.verbose
    if old_verbose>0:
        learner.verbose = 0

    # Determine type of search
    if type_of_search == 'random':
        generator = ParameterSampler(param_grid, n_iter=n_iter)
        number_of_fits = n_iter

    elif type_of_search == 'grid':
        generator = get_dicts(param_grid)
        number_of_fits = len(list(get_dicts(param_grid)))    
    
    else:
        learner.verbose = old_verbose
        raise ValueError("type_of_search should be in ('grid', 'random').")
    
    config, df = _search(learner=learner, name=name, generator=generator,
                        data=data, number_of_fits=number_of_fits,
                        modelfiles=modelfiles, top_n=top_n)
    
    learner.verbose = old_verbose    
    return config, df


def _search(learner, name, generator, data, number_of_fits, modelfiles, top_n):
    
    scores = []
    params_scores = [] 
    
    for param in tqdm_notebook(generator, desc=name, total=number_of_fits):        
        param.update(_train_eval(name, learner, data, param))
        params_scores.append(param) 
        
        # If no validation data, use train data to sort
        this_score = param['val_score'] 
        if np.isnan(this_score):
            this_score = param['train_score']
            
        rank = 0
        for score in scores:
            if score>=this_score: 
                rank += 1
                
        # If modelfiles is an empty list the below expression
        # never evaluates to True, and no files are saved.
        if rank<len(modelfiles):
            _save_model(learner.clf[name].estimator, modelfiles[rank])    
        scores.insert(rank, this_score)            
    
    return _get_result(params_scores, top_n)


def _get_result(params_scores, top_n):
    
    cols = ['train_loss', 'val_loss', 
            'train_score', 'val_score']
    
    df = (pd.DataFrame
          .from_dict(params_scores)
          .sort_values(['val_loss', 'val_score', 'train_loss', 'train_score'], 
                       ascending=[True, False, True, False]))  
    
    df = df[[c for c in df.columns 
             if not c in cols]+cols]
    
    config = []
    d = df.T.to_dict()
    for i in range(top_n):
        conf = d[df.index[i]].copy()
        for key in cols: del conf[key]
        config.append(conf)
        del conf
        
    return config, df