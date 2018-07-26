"""
Module implements functions and classes to help with bagging of predictions.

"""
import numpy as np
from sklearn.exceptions import NotFittedError


class RandomEnsembler:
    
    def __init__(self, nc_estimator, fitted_params, n_ensembles=5, seed=None):
        """
        Convenient way to average predictions from multiple instances of a single classifier 
        using the best parameters found from some type of hyperparameter search.
        
        Parameters:
        -----------
            nc_estimator : class template that implements random_state
                A class of scikit-learn estimator without closure (hence 'nc'=no closure)
                e.g., 'RandomForestClassifier' and *not* 'RandomForestClassifier()'
            
            fitted_params : dict
                Dictionary, a set of (preferably optimized) parameters to be fed to estimator.
            
            n_ensembles : integer, default: 5
                Control number of instances to ensemble over.
                
        """
        assert hasattr(nc_estimator(), 'random_state')
        self.nc_estimator = nc_estimator
        
        assert isinstance(fitted_params, dict)
        self.fitted_params = fitted_params        
        
        self.n_ensembles = int(n_ensembles)        
        if seed is not None:
            np.random.seed(seed)
        
        self.rargs = {'low': 1, 'high': 1e10}           
        self.models = []
    
    
    def get_models(self):
        """
        Returns the individual fitted models
        if `fit` has been called. 
        Otherwise return `None`.
        
        """
        if self.is_fitted:
            return self.models
        else:        
            return None
    
    
    def fit(self, X, y, **kwargs):
        """
        Parameters:
        -----------
            X : array-like, shape (n_samples, n_features). 
            Training data.
            
            y : array-like, shape (n_samples,). 
                Training labels.
            
            **kwargs : other parameters to input to `fit` method if applicable.
        """
        # Generate and train new model for each seed
        for _ in range(self.n_ensembles):           
            model = self.nc_estimator(
                random_state = np.random.randint(**self.rargs))
            model.set_params(**self.fitted_params)
            model.fit(X, y, **kwargs)
            self.models.append(model)
            del model    
        
        self.classes = np.unique(y)
        self.is_fitted = True               
        return
    
    
    def predict(self, X):
        """
        Generate class predictions using a majority voting
        scheme.
        
        Parameters:
        -----------
            X : matrix-like, shape (n_samples_test, n_features). 
                Test data to predict on.
                
        Returns:
        -----------
            ensembled_preds : numpy-array, shape (n_samples_test,). 
                Majority ensembled predictions which usually beat 
                the single best classifier predictions trained on 
                the same data.
                
        """    
        if not self.is_fitted:
            raise NotFittedError('Call `fit` first.')
        
        preds = np.zeros((X.shape[0], self.n_ensembles), dtype=int)
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)            

        # This is a trick that allows for 
        # vectorization of computations
        weights = np.ones(preds.shape[1])

        _classmapper = {}
        for i, cls in enumerate(self.classes):
            _classmapper[i] = cls
            belief = np.matmul(preds[:,:]==cls, weights).reshape(-1, 1)
            weighted = belief if i==0 else np.hstack((weighted, belief))
            
        predicted_class = map(lambda idx: _classmapper[idx], np.argmax(weighted, axis=1))            
        return np.array(list(predicted_class))   
 
    
    def predict_proba(self, X, get_classes=False):
        """
        Ensemble probability calculations.
        
        Parameters:
        -----------
            X : matrix-like, shape (n_samples, n_features). 
                Data to predict on.
            
            get_classes : boolean, default: False
                If True, convert raw probabilities to class 
                predictions instead, and output shape 
                transforms: (n_samples, n_classes) --> (n_samples,)
                
        Returns:
        -----------
            probas : numpy-array, shape (n_samples, n_classes)
                Numerically averaged probability predictions for each class which usually 
                beat the single best classifier predictions trained on the same data.
            - Note: if get_class=True, then probas is converted to an array of class
                predictions with shape (n_samples,)
                
        """  
        if not self.is_fitted:
            raise NotFittedError('Call `fit` first.')
            
        if not hasattr(self.nc_estimator(), 'predict_proba'):
            raise AttributeError('Estimator should implement `predict_proba`.')
        
        preds = np.ndarray((X.shape[0], len(self.classes), self.n_ensembles), dtype=float)
        for i, model in enumerate(self.models):
            preds[:, :, i] = model.predict_proba(X)            
        
        probas = np.average(preds, axis=-1, weights=None)        
        
        if get_classes:
            probas = self.proba_to_class(probas)
        return probas 
            
        
    def proba_to_class(self, probas):
        """ Convert probabilities to class prediction. """
        _classmapper = {}
        for i, cls in enumerate(self.classes):
            _classmapper[i] = cls
            
        predicted_class = map(lambda idx: _classmapper[idx], 
                              np.argmax(probas, axis=1))
        return np.array(list(predicted_class))   
