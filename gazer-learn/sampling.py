import numpy as np

class Loguniform(object):
    
    def __init__(self, low, high, size=1, base=10):
        """
        A Wrapper function that generates random variate samples from a log-uniform distribution between
        'low' and 'high'. The default base is set to 10, and a single sample is generated from loguniform().rvs()
        
        """
        self.a = low # DO NOT CHANGE (used in method 'skopt_space_mapping' in utils.py)
        self.b = high # DO NOT CHANGE (used in method 'skopt_space_mapping' in utils.py)        
        self._size = size
        self._base = base
        self.args = (self.a, self.b, self._size, self._base)
        
    def rvs(self, *args, **kwargs):
        if self._size == 1:
            return np.power(self._base, np.random.uniform(np.log10(self.a), np.log10(self.b)))
        else:
            return np.power(self._base, np.random.uniform(np.log10(self.a), np.log10(self.b), self._size))
    
    def range(self, *args, **kwargs):
        """
        Return a range of floats from 'low' to 'high' in powers of (almost always) 10.
        """
        return np.power(self._base, np.linspace(np.log10(self.a), np.log10(self.b), self._size, endpoint=True))