from __future__ import print_function

import os, sys, time, copy, glob, random, warnings
from operator import itemgetter
import numpy as np
from sklearn.externals import joblib 
from tqdm import tqdm_notebook as tqdm

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
        
        models : optional, dict, default: None
            Only used when instantiating from a pre-existing state. 
            Activated by from_state = True (see below).
            
          - Note: automatically computed by classmethod 'from_state'
            and passed into the constructor. You should never set 
            this variable manually: use 'GazerMetaEnsembler.from_state()'
            and pass the top-directory wherein your model files are located.
            
        from_state : bool, default: False
            Instantiate an ensembler from pre-existing files when True.
            The default behavior is to build a new ensemble from scratch
            by calling the internal '_build()' method.
    
    Notes: 
    ------
        
        >>> ensembler = GazerMetaEnsembler.from_state(files)
        # No need to perform fitting of models at this point
        # since we are loading from a state where this is taken care of.
        
        >>> ensembler.hillclimb()        
        # Instead, dive straight into hillclimbing: make sure that there is consistency
        # between the data you have previously trained on, and the validation set you
        # pass into the hillclimbing method.

        
    """
    def __init__(self, learner, data_shape, models=None, from_state=False):

        self.learner = learner            
        if learner is not None:
            if not isinstance(learner, GazerMetaLearner().__class__):
                raise TypeError("learner must be a GazerMetaLearner.")
        
        self.data_shape = data_shape
        if data_shape is not None:
            if not isinstance(data_shape, tuple) and len(data_shape)==2:
                raise TypeError("data_shape must be a 2-tuple.")
            
        # These are set according to passed state variable
        if not from_state:
            self.ensemble = self._build()
            self.models = {}   
            self.allow_train = True
        elif from_state:
            self.ensemble = None
            self.models = models    
            self.allow_train = False
    
    
    @classmethod
    def from_state(cls, topdir):        
        kwargs = {'learner': None, 'data_shape': None, 'from_state': True}
        kwargs.update({'models': cls.fetch_state_dict(topdir)})        
        return cls(**kwargs)

    
    @staticmethod
    def fetch_state_dict(topdir):
        d = {}
        assert os.path.isdir(topdir)
        search_tree = os.walk(topdir)
        _ = next(search_tree)
        
        for dirpath, dirnames, dirfiles in search_tree:
            if dirnames:
                raise Exception("Tree is too deep. Remove subdirs: {}".format(dirnames))
            if dirfiles:
                key = os.path.basename(dirpath)
                d[key] = dirfiles
            else:
                warnings.warn("Empty dir: {} (skipping)".format(dirpath))
        return d
    
    
    def _build(self):
        """ Build ensemble from base learners 
        contained in the `learner` object.        
        """
        lib = library_config(self.learner.names, *self.data_shape)        
        build = {}
        for name, grid in lib:            
            # Check metadata to determine if external
            info = self.learner.clf[name].get_info()
            is_external = info.get('external', False)    
            
            # Here we take care of external packages with their
            # own api 
            if is_external:                
                if name=='neuralnet':
                    build[name] = grid            
            else:
                build[name] = self._gen_templates(name, grid)            
        return build
    
    
    def _gen_templates(self, name, params):    
        """ Here we generate estimators to later fit. 
        """
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
        """ Generate a config grid. 
        """
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
        -----------
            X : matrix-like
                2D matrix of shape (n_samples, n_features)

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
        --------
            Dictionary with paths to fitted and pickled learners, as well as scores on 
            training data. Note that joblib is used to pickle the data.

        """
        if not self.allow_train:
            raise Exception("Loaded from existing state: training not possible. "+\
                "Try calling .hillclimb(X, y,..) method instead.")            
        if not save_dir:
            raise Exception("'{}' is not a valid directory.".format(save_dir))
        if os.path.exists(save_dir):
            warnings.warn("Warning: overwriting existing folder {}.".format(save_dir))
        else:
            os.makedirs(save_dir)
            
        self.models = self._fit(X=X, y=y, save_dir=save_dir, 
                                      scorer=get_scorer(scoring), 
                                      n_jobs=n_jobs, verbose=verbose, 
                                      **kwargs)
    
    def _fit(self, X, y, save_dir, scorer, n_jobs, verbose, **kwargs):
        """ Implement fitting. 
        """
        # Keep track of model and score
        # All relevant data is available in `history`
        history = {}
               
        names = list(self.ensemble.keys())
        for name in names:
            os.makedirs(os.path.join(save_dir, name))        
        
        name = 'neuralnet'
        if name in names:            
            args, param_grid = self.ensemble.pop(name)            
            n_iter = args['n_iter']            
            data = {'train': (X, y), 'val': None} 
            modelfiles = [os.path.join(save_dir, name, file) for file in args['modelfiles']]            
            _, df = param_search(
                self.learner, param_grid, 
                data=data, 
                type_of_search='random', 
                n_iter=n_iter, 
                name=name, 
                modelfiles=modelfiles)
            history[name] = zip(modelfiles, df.head(len(modelfiles))['train_score'].values)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, estimators in self.ensemble.items():                            
                path = os.path.join(save_dir, name)
                kwarg = kwargs.get(name, {})                                           
                if n_jobs != 1:
                    models = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend="threading")(
                        joblib.delayed(single_fit)(estimator, scorer, X, y, path, i, **kwarg) 
                        for i, estimator in enumerate(estimators, start=1))
                else:
                    models = []
                    for i, estimator in enumerate(tqdm(estimators, desc="{}".format(name), ncols=120)):
                        this_modelfile, this_score = single_fit(estimator, scorer, X, y, path, i, **kwarg)
                        models.append((this_modelfile, this_score))                   
                history[name] = sorted(list(models), key=lambda x: -x[1]) 
            
        # Purge any failed fits
        for name, models in history.items():
            history[name] = [(name, file) for file, _ in models if file is not None]            
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
   

    def hillclimb(self, X, y, n_best=0.1, p=0.3, iterations=10, scoring='accuracy', 
                  greater_is_better=True, n_jobs=1, verbose=0):
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
                
            scoring : str, default: 'accuracy'
                The metric to use when hillclimbing
            
            greater_is_better : boolean, default: True
                If True then a higher score on the validation
                set is better.
            
            n_jobs : int, default: 1
                Parallel processing of files.
                
            verbose : int, default: 0
                Whether to output extra information or not.
                - Set verbose = 1 to get info.               
        
        """
        if isinstance(n_best, float):
            grab = int(n_best*len(self.models))
        elif isinstance(n_best, int):
            grab = n_best
        else:
            raise TypeError("n_best should be int or float.")
            
        nets = [path for name, path in self.models if (name == 'neuralnet')]    
        clfs = [path for name, path in self.models if (name != 'neuralnet')]                
        scorer = get_scorer(scoring)        
        parallel = joblib.Parallel(n_jobs=n_jobs, 
                                   verbose=verbose, 
                                   backend="threading")
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')      
            sklearn = parallel(joblib.delayed(_sklearn_score_fitted)(path, X, y, scorer) 
                               for path in clfs) if clfs else []
            time.sleep(1)
            external = parallel(joblib.delayed(_keras_score_fitted)(path, X, y, scorer) 
                                for path in nets) if nets else []
        pooled = sorted([clf for clf in sklearn+external if not clf is None], 
                        key=itemgetter(2))                 
        del sklearn
        del external        
        
        if verbose > 0:        
            max_score = max(pooled, key=itemgetter(2))
            print("Single model max validation score = {}".format(np.round(max_score, 4))) 
        
        pooled = [(str(idx), clf, preds) for idx, (clf, preds, _) in enumerate(pooled)]
        ensemble = pooled[:grab]        
        weights = {idx: 0. for idx, *_ in pooled}    
        for idx, *_ in ensemble: weights[idx] = 1.
        
        if verbose > 0: 
            print("Best model: {}".format(ensemble[0][1]))                         
    
        all_ensembles = []
        for _ in range(iterations):
            this_ensemble = self._hillclimb_loop(X = X, y = y, scorer = scorer, ensemble = ensemble, 
                                        weights = weights, pooled = pooled, p = p, verbose = verbose)
            if this_ensemble: 
                all_ensembles.append(this_ensemble)
        
        scores = []
        ensembles = []
        for ensemble in all_ensembles:
            scores.append(ensemble[-1])
            ensembles.append(ensemble[:-1])        
        
        max_score = max(scores)        
        return max_score, ensembles[scores.index(max_score)]
    
    
    def _hillclimb_loop(self, X, y, scorer, ensemble, weights, pooled, p, verbose, seed=None):
        """ Execute hillclimb loop.        
        """ 
        max_iter = 100
        val_scores = []
        best_score = float("-Inf") if greater_is_better else float("Inf")
        if seed is not None: 
            np.random.seed(seed)
        scargs = {'greater_is_better': greater_is_better}
                
        hc_weights = weights.copy()
        hc_ensemble = ensemble.copy()
        hc_pool = self.sample_algorithms(p, pooled)

        curr_score = self.score(hc_ensemble, hc_weights, y, scorer)      
        if verbose > 0:
            print("Initial ensemble score = {:.4f}".format(curr_score))
                  
        for i in range(1, max_iter):                                    
            for algorithm in hc_pool:                
                idx = algorithm[0]                
                local_ensemble = hc_ensemble.copy()
                local_ensemble.append(algorithm)                         
                local_weights = hc_weights.copy()
                local_weights[idx] += 1
                
                this_score = self.score(local_ensemble, local_weights, y, scorer)            
                if rank_scores(this_score, best_score, **scargs)
                    best_idx = idx
                    best_score = this_score
                    best_algorithm = [algorithm]
            
            if rank_scores(curr_score, best_score, strict=False, **scargs)
                print("Failed to improve. Updated score was: {:.4f}".format(best_score))
                break                        
            elif rank_scores(best_score, curr_score, **scargs)
                curr_score = best_score
                hc_weights[best_idx] += 1                
                if not best_idx in self.get_idx(hc_ensemble):
                    hc_ensemble += best_algorithm
            
            val_scores.append((i, curr_score))
            if verbose > 0:
                print("Loop iter: {} \tScore: {:.4f}".format(*val_scores[-1]))
            
        weighted_ensemble = [(path, hc_weights[idx]) for idx, path, _ in hc_ensemble]        
        weighted_ensemble.append(val_scores[-1][-1])
        return weighted_ensemble
        
    
    @staticmethod
    def rank_scores(score, score_to_compare, greater_is_better, strict=True):
    if strict:
        op = operator.gt if greater_is_better else operator.lt
    else:
        op = operator.ge if greater_is_better else operator.le
    return op(score, score_to_compare)
        
        
    def score(self, ensemble, weights, y, scorer):
        """ Compute weighted majority vote. 
        """        
        wts = np.zeros(len(ensemble))
        preds = np.zeros((len(y), len(ensemble)), dtype=int)        
        for col, (idx, _, pred) in enumerate(ensemble):
            wts[col] = weights[idx]
            preds[:, col] = pred           
        return self.weighted_vote_score(wts, preds, y, scorer)

    
    def weighted_vote_score(self, weights, preds, y, scorer):  
        """ Score an ensemble of classifiers using weighted voting. 
        """
        classes = np.unique(preds)
        classmapper = {}
        for i, cls in enumerate(classes):
            classmapper[i] = cls
            belief = np.matmul(preds[:,:]==cls, weights)
            weighted = belief if i==0 else np.vstack((weighted, belief))
        predicted_class = np.array(
            list(map(lambda idx: classmapper[idx], np.argmax(weighted.T, axis=1))))             
        return scorer(predicted_class, y)
    
    
    def sample_algorithms(self, p, pool):        
        """ Sample algorithms from repository
        """
        idxmapper = {idx: (idx, clf, pr) for idx, clf, pr in pool}               
        if isinstance(p, float):
            size = int(p * float(len(pool)))
        elif isinstance(p, int):
            size = p        
        return list(map(lambda idx: idxmapper[idx], 
                        np.random.choice(self.get_idx(pool), 
                                         size=size, replace=False)))
            
    def get_idx(self, item):
        return [idx for idx, *_ in item]