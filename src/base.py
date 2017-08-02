"""
Module contaning definintions of base Classes.
"""


class BaseClassifier():
    """Base class for all tunable classifiers"""
    
    def __init__(self):
        self.__classname__ = 'BaseClassifier'
    
    def trainable_hyperparams(self, params:dict, num_params:int=1, mode:str='random', keys:list=[]) -> dict:
        """
        Docstring:
        
        Input args:
        ---------------
                params: dictionary contaning all hyperparameters belonging to classifier
            num_params: integer specifying number of parameters to sample if mode=='random'
                  mode: string, 'random' (sample) or 'select' (choose)
                  keys: list of keys (subsample of keys contained in 'params'
                  
        Output:
        ---------------
            dictionary containing parameters to train
            
        """        
        if not isinstance(params, dict):
            raise TypeError("Expected 'dict' type. Got '%s'" % type(params)
        if not mode in ['random', 'select']:
            raise ValueError("mode should be 'random' or 'select'.")
        
        items = params.items()        
        
        if mode == 'random':
            import random                             
            if not 0<num_params<=len(items):
                raise ValueError("Expect 0 < num_params <= items in dict.")
            return dict(random.sample(items, num_params, replace=False))
        
        elif mode == 'select':
            if not 0<len(keys)<=len(items): 
                raise ValueError("Expect 0 < len(keys) <= items in dict.")
            par = {}
            for key, v in items:
                if key in keys: 
                    par[key] = v
            if not len(par)>0: 
                raise Exception("No trainable parameters found: check 'keys' input.")
            return par
            
            
class EnsembleBaseClassifier():
    def __init__(self):
        pass
    
    def set_base_clf(self, clf):
        pass
    
    
class BaseRegressor():
    def __init__(self):
        raise NotImplementedError("Base class not yet implemented")
class EnsembleBaseRegressor():
    def __init__(self):
        raise NotImplementedError("Base class not yet implemented")
    
if __name__ == '__main__':
    import sys
    sys.exit(-1)