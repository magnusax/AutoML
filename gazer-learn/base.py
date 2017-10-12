"""
Module containing definitions of base Classes.
"""
import random  
import warnings

class BaseClassifier():
    
    def __init__(self):
        """Base class for all tunable classifiers"""
        self.__classname__ = 'BaseClassifier'
    
    def adjust_params(self, parms):
        """ Adjust classifier parameter helper function """

        if not isinstance(parms, dict):
            raise ValueError("Expect 'dict'. Got '%s'" % type(parms))
        fail = 0    
        for k, v in parms.items():
            try: 
                self.estimator.set_params(**{k:v})
            except: 
                fail =+ 1
        if fail > 0:
            warnings.warn("warning: at least one parameter not set.")
        return self
    
    def freeze_cv_params(self, parms):
        """ Freeze a trainable parameter to fixed value. Useful for when grid searching """
        if not instancde(parms, dict):
            raise ValueError("Expect 'dict'. Got '%s'" % type(parms))
        for k, v in parms.items():
            if k in self.estimator.cv_params.keys():
                self.estimator.cv_params[k] = v
        return self
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=list()):
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
            raise TypeError("Expected 'dict' type. Got '%s'" % type(params))
        if not mode in ('random', 'select'):
            raise ValueError("mode should be 'random' or 'select'")
        items = params.items()        
        
        if mode == 'random':                           
            if not 0<num_params<=len(items):
                raise ValueError("Expect 0 < num_params <= items in dict.")
            return dict(random.sample(items, num_params))
        
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
        self.__classname__ = 'EnsembleBaseClassifier'
    
    def set_base_estimator(self, clf):
        pass
    
    def set_train_params(self, params, num_params=1, mode='random', keys=[]):
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
            raise TypeError("Expected 'dict' type. Got '%s'" % type(params))
        if not mode in ['random', 'select']:
            raise ValueError("mode should be 'random' or 'select'.")        
        items = params.items()        
        
        if mode == 'random':                           
            if not 0<num_params<=len(items):
                raise ValueError("Expect 0 < num_params <= items in dict.")
            return dict(random.sample(items, num_params))
        
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
    
    
    
class BaseRegressor():
    def __init__(self):
        raise NotImplementedError("Base class not yet implemented")

class EnsembleBaseRegressor():    
    def __init__(self):
        raise NotImplementedError("Base class not yet implemented")