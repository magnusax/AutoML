"""
Utility module where convenience methods are placed
"""
import numpy as np
from skopt.space import Real, Categorical, Integer  


def skopt_space_mapping(name_params):
    """
    This method maps a dictionary of parameter settings to the appropriate scikit-optimize format     
    Input:
    -------------
    name_params: an iterable of (classifier name, classifier parameter dictionary) tuples   
    Ouput:
    -------------
    A list of (classifier name, classifier dictionary in skopt space format)    
    """         
    dicts = []
    # If an iterable has 100 elements or more, we convert it to either discrete (Integer dist.)
    # or continuous (Real dist.) distribution, with values between min(iterable) and max(iterable)
    # This only happens when we see 1 single data type in the iterable
    __THRESHOLD__ = 100
    
    for name, params in name_params:
        res = {}
        if len(params) > 0:
            for k, iterable in params.items():
                
                # check if is a sampling distribution
                if hasattr(iterable, 'rvs'):
                    min_ = iterable.a
                    max_ = iterable.b 
                    test_val = iterable.rvs()
                    if isinstance(test_val, float):
                        res[k] = Real(min_, max_)
                    else:
                        res[k] = Integer(min_, max_+1)
                    continue

                data = [str(type(x)) for x in iterable]
                n_data_types = len(np.unique(data))
                
                # check if 'iterable' really should be represented as a continous distribution
                # despite not having the rvs property.
                if (n_data_types == 1) and (len(data) >= __THRESHOLD__):
                    test_val = iterable[0]                    
                    try:
                        min_ = min(iterable)
                        max_ = max(iterable)
                    except:
                        res[k] = Categorical(list(iterable))
                    else:
                        if isinstance(test_val, float):
                            min_ = 10**np.floor(np.log10(min_))
                            max_ = 10**np.ceil(np.log10(max_))
                            res[k] = Real(min_, max_, prior='log-uniform')
                        elif isinstance(test_val, int):
                            res[k] = Integer(min_, max_)
                        else:
                            res[k] = Categorical(list(iterable))
                    continue
                    
                # if we get here, then 'iterable' definitely should be categorical
                res[k] = Categorical(list(iterable))                
                
        dicts.append((name, res))    
    return dicts


def one_of_k_encoding(cols, df):
    """ Categorial encoding of columns where datatype is categorical. 
        Input:    cols [list]
                  df   [pd.DataFrame]
        Output:   updated df
    """
    assert type(cols)==list, "Expecting a list."
    for col in cols:
        if col in df.columns.values: # Silently skip if not present
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df