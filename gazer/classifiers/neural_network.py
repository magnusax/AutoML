from __future__ import division
from __future__ import print_function

from ..base import BaseClassifier
from ..externals.clr_callback import CyclicLR

import os
import warnings
import collections
import numpy as np
import pandas as pd

import keras
import keras.backend as K
from keras.models import Sequential

from keras.callbacks import (
        EarlyStopping,
        LearningRateScheduler, 
        ReduceLROnPlateau, 
        ModelCheckpoint)

from keras.layers import (
        Activation, 
        BatchNormalization, 
        Dense, Dropout)


class MetaNeuralNetworkClassifier(BaseClassifier):
    
    def __init__(self, epochs=50, batch_size=32, optimizer='adam', learning_rate=1e-3, 
                 n_hidden=2, p=0.1, dropout=True, batch_norm=False, decay_units=False, input_units=250):
    
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
                
            input_units : integer, default: 250
                The number of units (neurons) to use in the input layer.
                
        """
        self.name = 'neuralnet'
        
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
        
        self.network['decay_units'] = decay_units
        # Handle neuron distribution per layer
        if self.network['decay_units']:
            self.network['units'] = [max(1, int(input_units//2**i)) for i in range(n_hidden+1)]
        else:
            self.network['units'] = [input_units] * (n_hidden+1)
            
        # Callbacks placeholder
        self.network['callbacks'] = self._callbacks()
        
        # Estimator is yet to be defined at this stage
        self.estimator = None
        self.ready = False
    
    
    def get_info(self):
        return {'does_classification': True,
                'does_multiclass': True,
                'does_regression': False, 
                'predict_probas': True}
    
            
    def set_architecture(self, input_shape, output_shape):
        """
        Called prior to building and compiling the keras model.
        """
        # Handle different representations
        if isinstance(input_shape, collections.Iterable):
            self.input_shape = (*input_shape,)
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
        """
        Implement callbacks.
        """
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50)
        
        # Checkpointing
        filepath = os.path.join(os.getcwd(), "tmp")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, "weights.{epoch:02d}_{loss:.2f}.hdf5")              
        checkpointing = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, 
                                        save_weights_only=True, period=10)           
        
        return[reduce_lr, early_stop]
    
    
    def lr_finder(self, X, y, monitor='acc', step_size=2000):
        """
        Find optimal learning rate
        
        """
        if not monitor in ('acc', 'loss'):
            raise ValueError("monitor should be either 'acc' or 'loss'.")
            
        if step_size is None:
            # Authors recommend step_size = 2-8 * iterations_per_epoch
            step_size = 8 * int(np.ceil(X.shape[0]/self.network['batch_size']))
        clr = CyclicLR(base_lr=1e-6, max_lr=10.0, 
                       step_size=step_size, mode='triangular', lr_find=True)
        
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
    
    
    def fit(self, X, y, **kwargs):
        
        # We keep a local copy of the labels 
        # to avoid modifying the original input data
        y_ = y.copy()
        
        if len(y_.shape)==1: y_ = y_.reshape(-1, 1)
        
        if y_.shape[1]==1:
            warnings.warn(
                """Keras expects one-hot encoded label data: your data does not seem to fit this requirement.
                   \nWill attempt to apply one-hot encoding before sending to `fit` method.""")
            y_ = keras.utils.to_categorical(y_)
        
        # This will only happen once as `set_architecture` modifies
        # this variable. Successive calls to `fit` will not cause
        # a recompilation of the model (i.e. you don't need to start from scratch!)
        if self.ready == False:
            # Freeze architecture
            self.set_architecture(X.shape[1], y_.shape[1])
            
            # Define estimator and compile keras model
            self.estimator = self._get_clf()
        
        self.network['history'] = \
            self.estimator.fit(
                X, y_, 
                batch_size = self.network['batch_size'], 
                epochs = self.network['epochs'],
                callbacks = self.network['callbacks'],
                verbose = 0,
                **kwargs)     
