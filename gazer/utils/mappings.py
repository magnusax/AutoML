from scipy.stats import uniform
import numpy as np
from skopt.space import (Real, 
                         Categorical, 
                         Integer) 


def skopt_space_mapping(all_params, threshold=100):
    """
    This method maps a dictionary of parameter iterables to the 
    appropriate 'scikit-optimize' format. This in order to be able to
    perform optimization using a Bayesian strategy.
    
    
    Parameters:
    ------------
        all_params : iterable, non-optional: List[tuple(), tuple(),..]
        
            Pass an iterable of (name, params) tuples where
            'name' corresponds to an initialized algorithm, and
            'params' is a dictionary of param_name keys. Their
            corresponding values should be iterable/scipy-distributions.            
            - Note: currently only scipy.stats.randint and scpipy.stats.uniform
                are supported.
                
        threshold : int, default: 100
            If an iterable does not have the 'rvs' property and is an iterable
            of length>threshold, consisting of a single datatype we convert
            the iterable to either a continuous or discrete uniform distribution.
            The choice of distribution depends on its datatype.
            -Note: min/max are dervied from min(iterable) and max(iterable).
                
    Returns:
    ---------
        List of (name, params) tuples in a format that suits scikit-optimize.
        Format: List[tuple(name1, params1), tuple(name2, params2),..] 
    
    
    Example:
    ----------
    All the below input formats are valid:
    
    >>> import numpy as np
    >>> from gazer.utils.mappings import skopt_space_mapping
    >>> params = {
          'n_estimators': range(50, 550, 50),
          'learning_rate': uniform(0.01, 0.2),
          'max_depth': randint(2, 10, 1),
          'colsample_bytree': np.linspace(0.5, 1.0, 6, endpoint=True),
          'colsample_bylevel': list(np.linspace(0.5, 1.0, 6, endpoint=True)) }       
    >>> input = [('xgboost', params)]    
    >>> result = skopt_space_mapping(input)
      
    """         
    spaces = []    
    for name, params in all_params:
        
        space = {}
        for key, iterable in params.items():    
            
            # check if iterable has 'rvs' method
            if hasattr(iterable, 'rvs'):
                min_, max_ = iterable.args[:2]
                
                assert (isinstance(min_, (float, int)) 
                        & isinstance(max_, (float, int)))
                
                if max_ <= min_:
                    raise ValueError("Algorithm {}, param {}: iterable_min={}, and iterable_max={}"
                                     .format(name, key, min_, max_))
                
                val = iterable.rvs()                
                if isinstance(val, float):
                    try:
                        if (iterable.dist.__class__ 
                            == uniform.__class__):
                            prior = 'uniform'
                    except:   
                        prior = 'log-uniform'
                    space[key] = Real(min_, max_, prior=prior)
                
                elif isinstance(val, int):
                    space[key] = Integer(min_, max_)
                
                else:
                    raise TypeError("Algorithm {}, param {}: Invalid sample type: {}"
                                    .format(name, key, repr(type(val))))
                continue

            # Check if iterable should be represented as 
            # a continuous/discrete distribution despite  
            # missing rvs property
            ntypes = len(np.unique(map(lambda x: type(x), iterable)))
            if (ntypes == 1) and (len(iterable) >= threshold):
                try:
                    min_, max_ = min(iterable), max(iterable)
                except:
                    space[key] = Categorical(list(iterable))
                else:
                    val = iterable[0]                                        
                    if isinstance(val, float):
                        space[key] = Real(min_, max_, prior='uniform')                    
                    elif isinstance(val, int):
                        space[key] = Integer(min_, max_)                   
                    else:
                        space[key] = Categorical(list(iterable))
                continue
                
            # if we get to this point in the loop, then 
            # 'iterable' definitely should be categorical
            space[key] = Categorical(list(iterable))                
        
        spaces.append((name, space))    
    return spaces
