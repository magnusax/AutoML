from __future__ import print_function

import os, sys, time, copy, glob, random, warnings

import numpy as np
from scipy.stats import uniform
from sklearn.externals import joblib 
from tqdm import tqdm_notebook

from .metrics import get_scorer
from .sampling import Loguniform
from .core import GazerMetaLearner
from .library import library_config
from .optimization import param_search


def single_fit(estimator, scorer, X, y, path, i, **kwargs):
    modelfile = os.path.join(path, "model_{:04d}train.pkl".format(i))
    try:
        estimator.set_params(**kwargs).fit(X, y)
        joblib.dump(estimator, modelfile)
        return (modelfile, scorer(estimator.predict(X), y))
    except:
        fail = (None, float("-Inf"))
        _, desc, _ = sys.exc_info()
        warnings.warn("Could not fit and save: {}".format(desc))
        return fail
    
    
def _sklearn_score_fitted(path, X, y, scorer):
    try:
        model = joblib.load(path)
        yhat = model.predict(X) 
        score = scorer(yhat, y)        
        return (path, yhat, score)
    except:
        return None
    
    
def _keras_score_fitted(path, X, y, scorer):
    """ Load previously fitted keras model. Then predict
    on `X` and return score based on comparison to `y`.
    """
    from keras.models import load_model
    import tensorflow as tf
    config = tf.ConfigProto()
    graph  = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph, config=config)
        with sess.as_default():
            try:
                model = load_model(path)
                yhat = model.predict_classes(X)
                score = scorer(yhat, y)
                return (path, yhat, score)
            except:
                return None
            
            
class GazerMetaEnsembler(object):
    """
    Ensembler class.

    Parameters:
    ------------
        learner : instance of GazerMetaLearner class
            Used to infer which algorithms to include in the 
            ensembling procedure

        data_shape : tuple, length 2
            Should specify input data dimensions according to
            (X.shape[0], X.shape[1]) where `X` is the canonical data-matrix
            with shape (n_samples, n_features)
    
    """
    def __init__(self, learner, data_shape):

        if not isinstance(data_shape, tuple) and len(data_shape)==2:
            raise TypeError("data_shape must be a 2-tuple.")
        self.data_shape = data_shape
        
        if not isinstance(learner, type(GazerMetaLearner())):
            raise TypeError("learner must be a GazerMetaLearner instance.")
        self.learner = learner
    
        # This object is later used to orchestrate 
        # hillclimbing on the validation dataset
        self.orchestrator = {}
        
        # Build ensemble dictionary
        self.ensemble = self._build()

        
    def summary(self):
        """ Summarize expected number of fits (individual and total). """
        total = 0
        for k, v in self.ensemble.items():
            total += len(v)
            print("Algorithm: {} \tFits: {}".format(k, len(v)))
        print("Expected total number of fits = {}".format(total))
 

    def _build(self):
        """
        Build ensemble from base learners contained in the `learner` object.

        Parameters:
        -----------
            learner : object
                instance of GazerMetaLearner class

            X : matrix-like
                input 2D matrix of shape (n_samples, n_features)
                We need some meta data to be able to make sensible choices
                on parameters.

        Returns:
        ---------
            Dictionary : (algorithm[str]: classifiers[list]) 
            Dictionary containing name keys with corresponding values being
            a list of possible learners with varying settings of hyperparameters.
        
        """
        lib = library_config(self.learner.names, *self.data_shape)        
        build = {}
        for name, grid in lib:
            
            # Check metadata: can we generate templates or not?
            info = self.learner.clf[name].get_info()
            external_api = info.get('external', False)    
            
            # Here we take care of algorithms from external packages
            # which we cannot follow the scikit-learn api 
            if external_api:                
                if name == 'neuralnet':
                    build[name] = grid            
            else:
                build[name] = self._gen_templates(name, grid)            
        return build
    
    
    def _gen_templates(self, name, params):    
        """ Here we generate estimators to later fit """
        
        clf = self.learner._get_algorithm(name)
        estimators = []
        for param in params:
            par = param['param']
            premise = param['config']
            values = self._gen_grid(param['grid'])        
            for value in values:
                estimator = copy.deepcopy(clf.estimator)
                pars = {par:value}
                pars.update(premise)
                try:
                    estimator.set_params(**pars)
                except:
                    warnings.warn("Failed to set {}".format(par))
                    continue
                estimators.append(estimator)
                del estimator                    
        return estimators

    
    def _gen_grid(self, grid):
        """ Generate a config grid. """
        method = grid.get('method', None)
        assert method in ('take', 'sample') 

        if method=='take':
            return grid['values']

        elif method=='sample':  
            category = grid.get('category', None)        
            assert category in ('discrete', 'continuous')
            
            low, high, points, prior = (
                grid['low'], grid['high'], grid['numval'], grid['prior'])

            if category=='discrete':
                raise NotImplementedError('Discrete sampling not implemented yet.')                   

            elif category=='continuous':                                  
                if prior=='loguniform':
                    return Loguniform(low=low, high=high, size=points).range()
                else:
                    return np.linspace(low, high, points, endpoint=True)
    
    
    def fit(self, X, y, save_dir, scoring='accuracy', n_jobs=1, verbose=0, **kwargs):
        """
        Fit an ensemble of algorithms.
        
        - Models are pickled under the `save_dir`
          folder (each algorithm will have a separate folder in the tree)
        - If directory does not exist, we attempt to create it. 

        Parameters:
        ------------
            X : matrix-like
                2D matrix of shape (n_samples, n_columns)

            y : array-like
                Label vector of shape (n_samples,)
            
            save_dir : str
                A valid folder wherein pickled algorithms will be saved
                
            scoring : str or callable
                Used when obtaining training data score
                Fetches get_scorer() from local metrics.py module
            
            n_jobs : integer, default: 1
                If n_jobs > 1 we use parallel processing to fit and save
                scikit-learn models. 
                Note: it is not used when training the neural network.
                
            verbose : integer, default: 0
                Control verbosity during training process.
                
            **kwargs: 
                Variables related to scikit-learn estimator.
                Used to alter estimator parameters if needed (such as e.g. n_jobs)

                Example: 
                    - Use e.g. {'random_forest': {'n_jobs': 4}} to use parallel
                      processing when fitting the random forest algorithm. 
                    - Note that the key needs to match the a key in the `ensemble` dict
                      to take effect. 
                    - The change takes place through estimator.set_params()

        Returns:
        ---------
            Dictionary with paths to fitted and pickled learners, as well as scores on 
            training data. Note that joblib is used to pickle the data.

        """         
        if (save_dir is None or len(save_dir)==0):
            raise Exception("Please specify a valid directory.")

        if os.path.exists(save_dir):
            warnings.warn("Warning: overwriting existing folder {}."
                          .format(save_dir))
        else:
            try:
                os.makedirs(save_dir)
            except:
                raise Exception("Could not create folder {}."
                                .format(save_dir))

        self.orchestrator = self._fit(X=X, y=y, save_dir=save_dir, 
                                      scorer=get_scorer(scoring), 
                                      n_jobs=n_jobs, verbose=verbose, 
                                      **kwargs)
        return
    
        
    def _fit(self, X, y, save_dir, scorer, n_jobs, verbose, **kwargs):
        """ Implement the fitting """
        
        # Keep track of model and scores
        # All relevant data is available in `history`
        history = {}
               
        names = list(self.ensemble.keys())
        for name in names:
            os.makedirs(os.path.join(save_dir, name))        
        
        # Ttrain the network first, if present
        _name = 'neuralnet'
        if _name in names:
            
            args, param_grid = self.ensemble.pop(_name)            
            _modelfiles = [os.path.join(save_dir, _name, file) 
                           for file in args['modelfiles']]
            _n_iter = args['n_iter']            
            _data={'train': (X, y), 'val': None}  
            # Do parameter search (random). Note: no validation data, so sort by train scores
            _, df = param_search(self.learner, param_grid, data=_data, type_of_search='random', 
                 n_iter=_n_iter, name=_name, modelfiles=_modelfiles)
            
            history[_name] = zip(_modelfiles, df.head(len(_modelfiles))['train_score'].values)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, estimators in self.ensemble.items():            
                
                path = os.path.join(save_dir, name)
                kwargs_ = kwargs.get(name, {})                           
                
                if n_jobs != 1:
                    models = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend="threading")(
                        joblib.delayed(single_fit)(estimator, scorer, X, y, path, i, **kwargs_) 
                        for i, estimator in enumerate(estimators, start=1))
                else:
                    models = []
                    for i, estimator in enumerate(tqdm_notebook(estimators, desc="{}".format(name), ncols=120)):
                        this_modelfile, this_score = single_fit(estimator, scorer, X, y, path, i, **kwargs_)
                        models.append((this_modelfile, this_score))
                    
                history[name] = sorted(list(models), key=lambda x: -x[1]) 
            
        # Before returning: purge any failed fits
        for name, models in history.items():
            history[name] = [(file, score) for file, score 
                             in models if file is not None]            
        return history
    
    
    def _add_networks(self, clf, X, y, path):
        """Add to ensemble repository a set of keras neural network
        models
        
        """
        y_ = clf.y_check(y, verbose=0)
        
        # Prepare for ensembling
        os.makedirs(path)
        clf.set_param('chkpnt_dir', path) 
        
        # When 'ensemble' is set to True, 
        # checkpointing to the 'path' folder is enabled
        clf.set_param('ensemble', True) 
        
        # Train
        print("Training neural net..")
                
        start = time.time()
        clf.fit(X, y_, verbose=0)
                
        print("Train time: {:.2f} min"
              .format((time.time()-start)/60.))
        time.sleep(1)
        
        # Evaluate and save 
        patterns = ('*.hdf5','*.h5','*.h5py')
        weightfiles = []
        for pattern in patterns:
            weightfiles += glob.glob(os.path.join(path, pattern))
        
        model = clf.estimator
        models = []
               
        for weightfile in tqdm(weightfiles, desc="Net (save wts)", ncols=120): 
            model.load_weights(weightfile)
            loss, score = model.evaluate(X, y_, verbose=0)
            models.append((weightfile, np.round(loss, decimals=4)))
        del y_, model
        
        # We sort according to loss: lower is better
        return (clf, sorted(models, key=lambda x: x[1]))
   
    
    def _prep_output(self):
        """
        Internal convenience method. Take object containing fitted algorithms 
        and scores and return a sorted list of classifiers.            
        """
        # Collect and order natives    
        files = []
        for name, fs in self.orchestrator.items():
            fs = [(name, path) for path, _ in fs]
            files += fs 
        return files
    
    
    def hillclimb(self, X, y, n_best=0.1, p=0.3, iterations=10, scoring='accuracy', n_jobs=1, verbose=0):
        """
        Perform hillclimbing on the validation data
        
        Parameters:
        ------------
            X : validation data, shape (n_samples, n_features)
            
            y : validation labels, shape (n_samples,)
            
            n_best : int or float, default: 0.1
                Specify number (int) or fraction (float) of classifiers
                to use as initial ensemble. The best will be chosen.
                
            p : float, default: 0.3
                Fraction of classifiers to select for bootstrap
                
            iterations : int, default: 10
                Number of separate hillclimb loop iterations to perform
                Due to the stochastic nature of the ensemble selection
                we try 'iterations' times to find the best one
                
            scoring : str, default: accuracy
                The metric to use when hillclimbing
            
            n_jobs : int, default: 1
                Parallel processing of files.
                
            verbose : int, default: 0
                Whether to output extra information or not.
                - Set verbose = 1 to get info.               
        
        """
        clfs = self._prep_output()        
        total = len([c for c,_ in clfs])

        if isinstance(n_best, float):
            grab = int(n_best*total)
        elif isinstance(n_best, int):
            grab = n_best
        else:
            raise TypeError("n_best should be int or float.")
            
        nets = [(name, path) for name, path in clfs 
                if (name == 'neuralnet')]    
        clfs = [(name, path) for name, path in clfs 
                if (name != 'neuralnet')]        
        
        scorer = get_scorer(scoring)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')      
            algs = joblib.Parallel(n_jobs=n_jobs, 
                                   verbose=verbose, 
                                   backend="threading")(
                joblib.delayed(_sklearn_score_fitted)(path, X, y, scorer) 
                for name, path in clfs) if clfs else []
            time.sleep(1)
            nalgs = joblib.Parallel(n_jobs=n_jobs, 
                                    verbose=verbose, 
                                    backend="threading")(
                joblib.delayed(_keras_score_fitted)(path, X, y, scorer) 
                for name, path in nets) if nets else []

        pool = [al for al in algs+nalgs if not al is None]          
        algs=None; clfs=None; nets=None; nalgs=None        
        
        if verbose > 0:        
            print("Single model max validation score = {}"
                   .format(np.round(max([score for *_, score in pool]), decimals=4)))  
        
        pool = [(str(idx), clf, y_pred) for idx, (clf, y_pred, _) 
                in enumerate(sorted(pool, key=lambda x: -x[2]))]                    
        ensemble = pool[:grab]        
    
        weights = {idx: 0.0 for idx, *_ in pool}    
        for idx, *_ in ensemble:
            weights[idx] = 1.0
        
        if verbose > 0:
            print("Best model was: {}".format(ensemble[0][1]))                         
    
        result = []
        for _ in range(iterations):
            # NB: do we wish to control the random seed here? (Reproducibility, etc.)
            wens = self._hillclimb_loop(X=X, y=y, scorer=scorer, ensemble=ensemble, 
                                        weights=weights, pool=pool, p=p, verbose=verbose)
            if wens: result.append(wens)
        
        scores = []
        ensembles = []
        for ensemble in result:
            scores.append(ensemble[-1])
            ensembles.append(ensemble[:-1])        
        max_score = max(scores)
        
        return max_score, ensembles[scores.index(max_score)]
    
    
    def _hillclimb_loop(self, X, y, scorer, ensemble, weights, pool, p, verbose, seed=None):
        """ Hillclimb loop. 
        
        """      
        __ensemble__ = ensemble.copy()
        __weights__ = weights.copy()
        __curr_score__ = self.score(__ensemble__, __weights__, y, scorer)
       
        if verbose > 0:
            print("Initial ensemble score = {:.5f}".format(__curr_score__))
        
        if seed is not None:
            np.random.seed(seed)           
        __pool__ = self._sample_algs(p, pool)
    
        # In practice we never max out
        for i in range(1, 100):            
            
            if i==1:
                best_score = float("-Inf")
                best_idx = -1
                val_scores = []
            
            for alg in __pool__:                
                idx = alg[0]                
                ens = __ensemble__.copy()
                ens.append(alg)                         
                wts = __weights__.copy()
                wts[idx] += 1.0
                
                this_score = self.score(ens, wts, y, scorer)            
                if this_score > best_score:
                    best_idx = idx
                    best_score = this_score
                    best_alg = [alg]
            
            if best_score <= __curr_score__:
                print("Could not improve further. Updated score was: {:.5f}"
                       .format(best_score))
                break            
            
            elif best_score > __curr_score__: 
                __curr_score__ = best_score
                __weights__[best_idx] += 1.0                
                if not best_idx in self._get_idx(__ensemble__):
                    __ensemble__ += best_alg
            
            val_scores.append((i, __curr_score__))
            if verbose > 0:
                print("Ensemble loop iter: {} \tScore: {:.5f}"
                      .format(*val_scores[-1]))
            
        weighted_ensemble = [
            (path_to_model, __weights__[idx]) 
            for idx, path_to_model, _ in __ensemble__]
        
        weighted_ensemble.append(val_scores[-1][-1])
        
        return weighted_ensemble
        
        
    def score(self, ensemble, weights, y, scorer):
        """ Compute weighted majority vote. 
        """        
        wts = np.zeros(len(ensemble))
        preds = np.zeros((len(y), len(ensemble)), dtype=int)        
        for col, (idx, _, pred) in enumerate(ensemble):
            wts[col] = weights[idx]
            preds[:, col] = pred
            
        return self._weighted_vote_score(wts, preds, y, scorer)

    
    def _weighted_vote_score(self, wts, preds, y, scorer):  
        """ Score an ensemble of classifiers using weighted voting. 
        """
        classes = np.unique(preds)
        _classmapper = {}
        for i, cls in enumerate(classes):
            _classmapper[i] = cls
            belief = np.matmul(preds[:,:]==cls, wts)
            weighted = belief if i==0 else np.vstack((weighted, belief))
        predicted_class = np.array(
            list(map(lambda idx: _classmapper[idx], np.argmax(weighted.T, axis=1))))             
        return scorer(predicted_class, y)
    
    
    def _sample_algs(self, p, pool):        
        _idxmapper = {idx: (idx, clf, pr) for idx, clf, pr in pool}               
        if isinstance(p, float):
            size = int(p * float(len(pool)))
        elif isinstance(p, int):
            size = p        
        return list(map(lambda idx: _idxmapper[idx], np.random.choice(self._get_idx(pool), 
                                                               size=size, replace=False)))
        #return [idxs_mapper[idx] for idx in 
        #        np.random.choice(self._get_idx(pool), 
        #                         size=size, replace=False)]
         
            
    def _get_idx(self, item):
        return [idx for idx, *_ in item]