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
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.externals import joblib


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
    train = learner.evaluate(*train_data, get_loss=True)[name]
    val = learner.evaluate(*val_data, get_loss=True)[name]
      
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

                 
def grid_search(learner, params, data, name='neuralnet', modelfiles=[]):
    """
    Grid search method customized for use with the Neural Network
    meta estimator, but works for any meta estimator in this library.
    
    - For native scikit-learn algorithms it is recommended to use GridSearchCV
    class instead. This method does e.g. not implement distributed fitting. 
        
    Parameters:
    ------------
        learner : instance of GazerMetaLearner
            - Grid search uses several learner instance methods in order to
            update parameters properly.
        
        name : str, 
            - A meta estimator identifier.
            - Must be present in learner.names.
        
        params : dict
            - A dictionary of parameters and corresponding values/iterables.
        
        data : dict
            - A dictionary containing train and validation data in the following format:
            data = {'train': (X_train, y_train), 'val': (X_val, y_val)}
            
        modelfiles : list, default: empty list
            - List with N elements, where N are names of the top N models
            that should be saved to disk.
            - If empty, no files are saved.
            
    Returns:
    ---------
    Tuple of (best) parameter configuration, and pandas dataframe with complete
    overview of parameters and their train + validation scores for easy comparison.
    
    - Note that the "best" configuration not necessarily corresponds to lowest 
    generalization error.
    
    """    
    # Keep silent during search
    old_verbose = learner.verbose
    if old_verbose>0:
        learner.verbose = 0

    scores = []
    params_scores = [] 
    number_of_fits = len(list(get_dicts(params)))     
    
    for param in tqdm_notebook(get_dicts(params), desc=name, total=number_of_fits):
        param.update(_train_eval(name, learner, data, param))
        params_scores.append(param) 
        
        this_score = param['val_score']        
        rank = 0
        for score in scores:
            if score>=this_score: 
                rank += 1
                
        # If modelfiles is an empty list the below expression
        # never evaluates to True, and no files are saved.
        if rank<len(modelfiles):
            _save_model(learner.clf[name].estimator, modelfiles[rank])    
        scores.insert(rank, this_score)
        
        
    # Restore when finished
    learner.verbose = old_verbose
    
    df = (pd.DataFrame
          .from_dict(params_scores)
          .sort_values('val_score', ascending=False))  
    
    config = df.T.to_dict()[df.index[0]]
    remove = ('val_loss', 'val_score', 
              'train_loss', 'train_score')
    for key in remove: del config[key]
          
    return config, df        
        
