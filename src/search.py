import numpy as np

class loguniform():
    
    def __init__(self, low, high, size=1, base=10):
        """
        A Wrapper function that generates random variate samples from a log-uniform distribution between
        'low' and 'high'. The default base is set to 10, and a single sample is generated from loguniform().rvs()
        
        """
        self.a = low # DO NOT CHANGE (used in method 'skopt_space_mapping' in utils.py)
        self.b = high # DO NOT CHANGE (used in method 'skopt_space_mapping' in utils.py)
        self._size = size
        self._base = base
        
    def rvs(self, *args, **kwargs):
        if self._size == 1:
            return np.power(self._base, np.random.uniform(np.log10(self._low), np.log10(self._high), self._size))[0]
        else:
            return np.power(self._base, np.random.uniform(np.log10(self._low), np.log10(self._high), self._size))
    

if __name__ == '__main__':
    import sys
    sys.exit(-1)