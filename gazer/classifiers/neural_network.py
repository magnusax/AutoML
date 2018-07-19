from __future__ import division
from __future__ import print_function

from ..base import BaseClassifier
from ..externals.clr_callback import CyclicLR

import os
import warnings
import collections
import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import (
        EarlyStopping,
        LearningRateScheduler, 
        ReduceLROnPlateau, 
        ModelCheckpoint)

from keras.layers import (
        Activation, 
        BatchNormalization, 
        Dense, 
        Dropout)


class MetaNeuralNetworkClassifier(BaseClassifier):
    """
    
    Parameters:
    ------------

        epochs : integer, default: 50
            The number of epochs to train for.

        batch_size : integer, default: 16
            Batch size used in the fit method.

        optimizer : string or keras callable. Default: str, 'adam'
            The optimizer to use when compiling. Note that 'learning_rate'
            overrides any learning rate that was set in the callable, so
            make sure the values match.

        learning_rate : float, default: 1e-3
            Specify learning rate to use in the optimizer. 

        n_hidden : integer, default: 2
            Specify the number of hidden dense layers.

        p : float, default: 0.1
            Specify dropout rate if applicable. 
            Only takes effect when 'dropout' is set to True or the dropout list contains
            at least one item which is set to True (see below).

        dropout : Boolean or list of booleans. Default: True.
            If providing a list: it must be of length n_hidden+1
            This parameter specifies the dropout per layer with rate given by 'p' (constant).

        batch_norm : Boolean or list of booleans. Default: False
            If providing a list: it must be of length n_hidden+1
            Here you can choose how to apply batch normalization to the architecture.       

        decay_units : boolean, default: False
            If False then keep the number of units constant at the value given by 'input_units'.
            If True then decay the number of units by a factor of 2 from layer to layer.
        
        gamma : float, default: 2.0
            If decay_units=True, then number of units in layer L+1 is decreased by
            a factor of gamma (constrained to lie in [1.5, 2.5]) compared to layer L.
            - Note: if gamma input is less than 1.5 then gamma=1.5, and if gamma>2.5 then
            gamma=2.5.
            
        input_units : integer, default: 250
            The number of units (neurons) to use in the input layer.

        chkpnt_dir : str, default: ''
            The directory to save model checkpoints (only weights are saved).
        
        validation_data, float, default: 0
            Used by keras fit api. If >0 then use a fraction of the data as
            validation data to estimate generalization error/overfitting.
            
    """
    def __init__(self, epochs=50, batch_size=32, optimizer='adam', learning_rate=1e-3, 
                 n_hidden=2, p=0.1, dropout=True, batch_norm=False, decay_units=False, 
                 gamma=2.0, input_units=250, chkpnt_dir='', chkpnt_per=1, validation_split=0.0):
    
        self.name = 'neuralnet'
        
        # Control callbacks (enable checkpointing, etc.)
        self.ensemble = False
        self.chkpnt_dir = chkpnt_dir
        self.chkpnt_per = chkpnt_per        
        self.validation_split = validation_split
        self.gamma = np.clip(gamma, 1.5, 2.5)        
        
        # We need to know the shape of the data, and the number of classes
        # They are set prior to calling `fit` through a call to `set_architecture`
        self.input_shape, self.output_shape = (None, None)        
        
        self.network = {}
        self.network['epochs'] = epochs
        self.network['batch_size'] = batch_size
        self.network['lr'] = learning_rate
        self.network['optimizer'] = optimizer
        self.network['history'] = None
        
        # Specify architecture
        self.network['n_hidden'] = n_hidden
        self.network['p'] = p
        self.network['activation'] = ['relu'] * (n_hidden+1)
        
        # Implement batchnorm and dropout per layer
        for key, var in zip(('batchnorm', 'dropout'), (batch_norm, dropout)):
            if isinstance(var, bool):
                self.network[key] = [var] * (n_hidden+1)
            elif isinstance(var, list):
                assert len(var)==(n_hidden+1)
                self.network[key] = var
            else:
                raise ValueError("Incorrect '%s' type." % key)
        
        # Handle neuron distribution per layer
        self.network['decay_units'] = decay_units
        
        if self.network['decay_units']:
            self.network['units'] = [max(1, int(input_units//self.gamma**i)) 
                                     for i in range(n_hidden+1)]
        else:
            self.network['units'] = [input_units] * (n_hidden+1)
            
        # Callbacks placeholder: set just before calling fit
        self.network['callbacks'] = []
        
        # Estimator is yet to be defined at this stage
        self.estimator = None
        self.ready = False
    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': True, 
                'external': True }
        
        
    def set_param(self, param, value):
        super().set_param(param, value)
               
            
    def _set_architecture(self, input_shape, output_shape):
        """Called prior to building and compiling the keras model. """
        
        # Handle miscellaneous representations
        if isinstance(input_shape, collections.Iterable):
            self.input_shape = input_shape
        else:
            self.input_shape = (input_shape,)
            
        # The `output_shape` is simply the number of class labels    
        self.output_shape = output_shape
        self.ready = True        
        return
       
        
    def _get_clf(self):

        """ 
        We use the sequential model API and introduce some flexibility
        in the number of hidden layers, hidden units, regularization, activation, 
        and learning rates.
        
        """         
        units = self.network['units']
        activations = self.network['activation']
        add_batchnorm = self.network['batchnorm']
        add_dropout = self.network['dropout']        
        p = self.network['p']
        
        model = Sequential()
        for i in range(self.network['n_hidden']+1):
            if i==0:
                model.add(Dense(units=units[i], input_shape=self.input_shape))
            else:
                model.add(Dense(units=units[i]))                
            model.add(Activation(activations[i]))            
            if add_batchnorm[i]: 
                model.add(BatchNormalization())                
            if add_dropout[i]: 
                model.add(Dropout(p))
            
        model.add(Dense(units=self.output_shape, activation='softmax'))                
        model.compile(
            loss='categorical_crossentropy',
            optimizer=self.network['optimizer'],
            metrics=['accuracy'])
        # Use the backend to set the base learning rate
        K.set_value(model.optimizer.lr, self.network['lr'])
        
        return model
        
        
    def check_training(X, y, train_size=0.1):
        """ 
        Perform training on p% (p<<100) of the training data to check that 
        we can include the algorithm without having to wait for a very long 
        time to finish training on the full dataset.
        """
        
        # We cap the size of the training fraction
        train_size = 0.5 if train_size > 0.5 else train_size
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=train_size)

        start_time = time.time()
        self.estimator.fit(X_sample, y_sample)
        total_time = time.time()-start_time
        
        # Output a warning 
        warnings.warn("Time spent training on %s%% of data: %.2f (min)" 
                                    % (100*train_size, total_time/60.))
                
    def _callbacks(self):
        """ Implement callbacks """
         
        monitor = 'val_loss' if self.validation_split>0 else 'loss'
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor, 
            factor=0.5, 
            patience=5, 
            min_lr=1e-5)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor=monitor, 
            min_delta=0.0001, 
            patience=20)
        
        # Checkpointing
        wtfile = 'weights.{epoch:02d}_{loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(os.path.join(self.chkpnt_dir, wtfile), 
            monitor=monitor, 
            save_best_only=False, 
            save_weights_only=True, 
            period=self.chkpnt_per)           
        
        if self.ensemble:
            return [reduce_lr, checkpoint,]
        else:
            return [reduce_lr, early_stop,]
    
    
    def lr_finder(self, X, y, monitor='acc', step_size=2000):
        """ Find optimal learning rate. """
        
        if not monitor in ('acc', 'loss'):
            raise ValueError("monitor should be either 'acc' or 'loss'.")
            
        if step_size is None:
            step_size = 8 * int(np.ceil(X.shape[0]/float(self.network['batch_size'])))
            
        clr = CyclicLR(base_lr=1e-6, 
                       max_lr=10.0, 
                       step_size=step_size, 
                       mode='triangular', 
                       lr_find=True)
        
        # Save previous values in in order to be 
        # able to restore when finished
        callbacks_old, epoch_old = \
            (self.network['callbacks'], self.network['epochs'])
        
        self.network['callbacks'], self.network['epochs'] = [clr], 10
        self.fit(X, y)
        
        clr = self.network['callbacks'][0]        
        ylr = pd.Series(clr.history[monitor]).rolling(window=3).mean()
        xlr = np.array(clr.history['lr'])
        
        # Restore
        self.network['callbacks'], self.network['epochs'] = \
            (callbacks_old, epoch_old)
        self.estimator = None
        self.ready = False
        
        return (xlr, ylr)
        
        
    def fit(self, X, y, verbose=0, **kwargs):
        """
        ### Copied from 'KerasClassifier' fit function ###
        Fit model to training data.
        
        Parameters:
        ------------
            X : array-like or matrix-like, shape (n_samples, n_features)
                Training data.
                
            y : array-like, shape (n_samples,)
                Training labels used as ground 
                truth to optimize.
                
            verbose : int, default: 0 (quiet)
                Optional verbosity parameter.
                
            **kwargs : dictionary
                Additional input arguments to 
                'Sequential.fit'.
        
        """
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif ((len(y.shape) == 2 and y.shape[1] == 1) 
              or len(y.shape) == 1):
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: {}'
                             .format(y.shape))       
        self.n_classes_ = len(self.classes_)
        
        if not self.ready:
            self._set_architecture(X.shape[1], 
                                   self.n_classes_)            
            self.estimator = self._get_clf()       
        
        loss_name = self.estimator.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if (loss_name == 'categorical_crossentropy' 
            and len(y.shape) != 2):
            y = to_categorical(y)
                
        self.network['history'] = self.estimator.fit(X, y, 
                                                     batch_size = self.network['batch_size'],
                                                     epochs = self.network['epochs'],
                                                     validation_split = self.validation_split,
                                                     callbacks = self.network['callbacks'],
                                                     verbose = verbose, **kwargs)
        return
    
    
    def predict(self, X, **kwargs):
        """
        ### Copied from 'KerasClassifier' predict function ###
        Returns the class predictions for the given test data.
        
        Parameters:
        ------------
            X : array-like, shape (n_samples, n_features)
                Test samples where n_samples is the number of samples
                and n_features is the number of features.

            **kwargs : dictionary arguments
                Legal arguments are the arguments
                of 'Sequential.predict_classes'.

        Returns:
        ---------
            preds : array-like, shape (n_samples,)
                Class predictions.
        """
        proba = self.estimator.predict(X, **kwargs)

        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]    