import pandas as pd
import numpy as np
from skopt.space import Real, Categorical, Integer  
from scipy.stats import uniform


def skopt_space_mapping(params):
    """
    This method maps a dictionary of parameter settings to the 
    appropriate scikit-optimize format     
    Input:
    -------------
    params: an iterable of (classifier name, 
            classifier parameter dictionary) tuples   
    
    Ouput:
    -------------
    List of (classifier name, classifier dictionary 
            in skopt space format)    
    """         
    spaces = []
    
    # If an iterable has 100 elements or more, we convert it to either discrete (Integer dist.)
    # or continuous (Real dist.) distribution, with values between min(iterable) and max(iterable)
    # This only happens when we see 1 single data type in the iterable
    threshold = 100
    
    for name, pars in params:
        space = {}
        for key, iterable in pars.items():            
            # check if 'iterable' can be sampled using the 'rvs' method
            if hasattr(iterable, 'rvs'):
                min_ = iterable.args[0]
                max_ = iterable.args[1]
                if max_ <= min_:
                    raise ValueError("Expected min < max. Found min=%s, max=%s" % (min_, max_))
                test_val = iterable.rvs()
                if isinstance(test_val, float):
                    try:
                        if iterable.dist.__class__ == type(uniform):
                            prior = 'uniform'
                    except:   
                        prior = 'log-uniform'
                    space[key] = Real(min_, max_, prior=prior)
                else:
                    space[key] = Integer(min_, max_)
                continue

            dtypes = [str(type(val)) for val in iterable]
            ntypes = len(np.unique(dtypes))
                
            # check if 'iterable' really should be represented as 
            # a continuous distribution despite not having the rvs property.
            if (ntypes == 1) and (len(dtypes) >= threshold):
                test_val = iterable[0]                    
                try:
                    min_ = min(iterable)
                    max_ = max(iterable)
                except:
                    space[key] = Categorical(list(iterable))
                else:
                    if isinstance(test_val, float):
                        min_ = 10**np.floor(np.log10(min_))
                        max_ = 10**np.ceil(np.log10(max_))
                        space[key] = Real(min_, max_, prior='uniform')
                    elif isinstance(test_val, int):
                        space[key] = Integer(min_, max_)
                    else:
                        space[key] = Categorical(list(iterable))
                continue                   
            # if we get to this point in the loop, then 'iterable' definitely should be categorical
            space[key] = Categorical(list(iterable))                
        spaces.append((name, space))    
    return spaces


def one_of_k_encoding(cols, df):
    """ 
    Categorial encoding of columns where datatype is categorical 
    Input:    cols [list], df [pd.DataFrame]
    Output:    updated df [pd.DataFrame]
    """
    for col in cols:
        if col in df.columns.values: # Silently skip if not present
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df