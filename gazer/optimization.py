"""

Credit:
--------
Methods iter_safe, get_dicts are from:
    https://codereview.stackexchange.com/questions/128088/
    convert-a-dictionary-of-iterables-into-an-iterator-of-dictionaries

"""

import os
import copy
import shutil
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import ParameterSampler

from .utils.meta import Mute
from .utils.estimators import save_model



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

        
def train_eval(name, learner, data, params):
    """ Update, train and evaluate. 
    """        
    learner.update(name, params)
    
    train_data, val_data = (data['train'], data['val'])
    learner.fit(*train_data)    
        
    kwargs = {'get_loss': True, 'verbose': 0}    
    train = learner.evaluate(*train_data, **kwargs)[name]    
    
    if val_data is None:
        val = {'loss': np.nan, 'score': np.nan}
    else:
        val = learner.evaluate(*val_data, **kwargs)[name]        
    
    return {'train_loss': train['loss'], 
            'train_score': train['score'],
            'val_loss': val['loss'], 
            'val_score': val['score']}


def get_result(params_scores, top_n):
    """ Extract CV results. 
    """
    cols = ['train_loss', 'val_loss', 'train_score', 'val_score']     
    df = (pd.DataFrame
          .from_dict(params_scores)
          .sort_values(['val_score', 'val_loss', 'train_score', 'train_loss'], 
                        ascending=[False, True, False, True]))    
    df = df[[c for c in df.columns if not c in cols]+cols]
    
    config = []
    d = df.T.to_dict()
    for i in range(top_n):
        conf = d[df.index[i]].copy()
        for key in cols: del conf[key]
        config.append(conf)
        del conf
        
    return config, df


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
            
        modelfiles : list or iterable of filenames, default: []
            Filenames wherein to save models.
            - Number of models to save is equal to len(modelfiles)
            
        top_n : integer, default: 1 
            Number of models to return in the model config dict.
            
    Returns:
    ---------
        Tuple of (best) parameter configuration(s), and pandas dataframe with complete
        overview of parameters and their train + validation scores for easy comparison.
    
    Notes:
    -------    
    - Number of parameter configurations set by 'top_n' parameter.   
    - Note that the "best" configuration not necessarily corresponds to lowest 
        generalization error.
    
    """    
    try:
        folder = os.path.dirname(modelfiles[0])
        shutil.rmtree(folder)
    except:
        pass    
    
    with Mute(learner):
        
        if type_of_search == 'random':
            generator = ParameterSampler(param_grid, n_iter=n_iter)
            number_of_fits = n_iter

        elif type_of_search == 'grid':
            generator = get_dicts(param_grid)
            number_of_fits = len(list(get_dicts(param_grid)))    

        else:
            raise ValueError("type_of_search should be in ('grid', 'random').")

        config, df = _search(learner=learner, name=name, generator=generator,
                            data=data, number_of_fits=number_of_fits,
                            modelfiles=modelfiles, top_n=top_n)    
    return config, df


def _search(learner, name, generator, data, number_of_fits, modelfiles, top_n):
    
    scores = []
    params_scores = [] 
    
    n_models = len(modelfiles)-1
    
    for params in tqdm(generator, desc=name, total=number_of_fits):        
        
        params.update(train_eval(name, learner, data, params))
        params_scores.append(param) 
        
        this_score = params['val_score'] 
        if np.isnan(this_score):
            this_score = params['train_score']
                
        if n_models > -1:       
            rank = len(np.where(np.array(scores) >= this_score)[0])            
            if rank < n_models:
                for _rank in reversed(range(rank, n_models)):                                        
                    src, dest = (modelfiles[_rank], 
                                 modelfiles[_rank+1])                                                                   
                    if os.path.exists(src):                                                
                        os.replace(src, dest)                
                scores.insert(rank, this_score)
                save_model(learner.clf[name].estimator, 
                           modelfiles[rank]) 
    
    return get_result(params_scores, top_n)