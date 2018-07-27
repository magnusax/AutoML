""" Module containing definitions of base Classes.

"""
import random  
import warnings


class BaseEstimator:
    """ Base class for estimators. 
    """    
    def __init__(self):
        self.__classname__ = 'BaseEstimator'
    
    def base(self, *args, **kwargs):
        raise NotImplementedError("'base' not implemented yet.")
        

        
class BaseClassifier(BaseEstimator):
    """ Base class for tunable classifiers. 
    """
    def __init__(self):
        self.__classname__ = 'BaseClassifier'    
    
    def set_params_meta(self, param, value):
        """ Adjust property of a meta estimator.
        """
        if not hasattr(self, param):
            raise Exception(
                "{} does not have property {}"
                .format(self.__class__.__name__, param))
        setattr(self, param, value)
        return self    
    
    def set_params(self, params):
        """ Adjust estimator parameters. 
        """                
        keys = params.keys()
        existing_keys = (self.estimator
                         .get_params().keys())       
        for key in keys:
            if not key in existing_keys:
                del params[key]                
        self.estimator.set_params(**params)
        return        
    
    def update_cv_params(self, params):
        """ Update cv params dictionary/ies with new parameters. """
        for d in self.cv_params:
            if not isinstance(d, dict):
                continue
            for k, v in params.items():
                if k in d.keys():
                    d.update(**{k:v})
        return    
    
    def freeze_cv_params(self, params):
        """ Freeze a trainable parameter to fixed value. 
        Useful for when grid searching. 
        """
        assert isinstance(params, dict)
        if self.cv_params:
            for d in self.cv_params:
                existing_keys = d.keys()
                for key, value in params.items():
                    if key in existing_keys:
                        d[key] = value
        else:
            warnings.warn("Empty 'cv_params'.")
        return 

    def set_tune_params(self, params, n_params, mode, keys):
        """
        Parameters:
        ------------
            params : dictionary containing all hyperparameters belonging to classifier

            n_params : integer specifying number of parameters to sample if mode = 'random'

            mode : str, values: 'random' (= sample), or 'select' (= choose)

            keys : list of keys (subsample of keys contained in params)

        Returns:
        ---------
            Dict with trainable parameters.
        
        """        
        if not mode in ('random', 'select'):
            raise ValueError("Require mode in ('random', 'select')")
        items = params.items()
                            
        if mode=='random':
            if not 0 < n_params <= len(items):
                raise ValueError("Bad 'n_params'.")    
            return dict(random.sample(items, n_params))

        elif mode=='select':
            if not keys and len(keys)>0:
                raise ValueError("Check 'keys'.")                            
            pars = {}
            for key, value in items:
                if key in keys:
                    pars[key] = value
            if not pars:
                raise Exception("Check 'keys': no trainables found.")
            return pars

        
class EnsembleBaseClassifier(BaseEstimator):    
    
    def __init__(self):
        self.__classname__ = 'EnsembleBaseClassifier'
    
    def set_base_estimator(self, clf):
        pass
    
    def set_tune_params(self, params, num_params=1, mode='random', keys=[]):
        """
        Parameters:
        -----------
                params: dictionary contaning all hyperparameters belonging to classifier
            num_params: integer specifying number of parameters to sample if mode=='random'
                  mode: string, 'random' (sample) or 'select' (choose)
                  keys: list of keys (subsample of keys contained in 'params'
                  
        Returns:
        -------
            Dictionary containing parameters to train            
        
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
    
    
##########################
# BASE REGRESSOR CLASSES
##########################


class BaseRegressor(BaseEstimator):    
    def __init__(self):
        raise NotImplementedError("Base class not yet implemented")
                            
class EnsembleBaseRegressor(BaseEstimator):    
    def __init__(self):
        raise NotImplementedError("Base class not yet implemented")

        