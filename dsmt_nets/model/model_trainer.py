"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import sys
import warnings
import numpy as np
import keras.callbacks as cbks
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.utils.data_utils import GeneratorEnqueuer

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


def main_generator_wrapper(validation_generator, steps, is_aux=False):
    def generator():
        while True:
            outputs = next(validation_generator)

            if len(outputs) == 2:
                x, (y, y_aux) = outputs
                sample_weights, sample_weights_aux = None, None
            else:
                x, (y, y_aux), sample_weights = outputs
                sample_weights, sample_weights_aux = sample_weights

            if is_aux:
                yield x, y_aux, sample_weights_aux
            else:
                yield x, y, sample_weights

    return generator()


class NoHDF5ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(NoHDF5ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        np.save(filepath, self.model.get_weights())
                        if not self.save_weights_only:
                            with open(filepath + ".json", "w") as text_file:
                                text_file.write(self.model.to_json())
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                np.save(filepath, self.model.get_weights())
                if not self.save_weights_only:
                    with open(filepath + ".json", "w") as text_file:
                        text_file.write(self.model.to_json())


class ModelTrainer(object):
    @staticmethod
    def train_dnn(model, train_set, validation_set, num_epochs, steps_per_epoch, validation_steps,
                  early_stopping_patience=12, with_checkpoints=True, do_eval=True, loss_name="val_loss",
                  do_early_stopping=True, checkpoint_path="./tmp/model.h5", save_best_only=True):
        show_output = 2
        callbacks = []

        if do_early_stopping:
            callbacks.append(EarlyStopping(monitor=loss_name,
                                           patience=early_stopping_patience))

        if with_checkpoints:
            callbacks.append(NoHDF5ModelCheckpoint(filepath=checkpoint_path,
                                                   monitor=loss_name,
                                                   save_best_only=save_best_only,
                                                   save_weights_only=False,
                                                   verbose=0))

        history = model.fit_generator(generator=train_set, epochs=num_epochs, steps_per_epoch=steps_per_epoch,
                                      validation_data=validation_set, validation_steps=validation_steps,
                                      callbacks=callbacks, verbose=show_output)

        if with_checkpoints:
            # Save model loss history.
            pickle.dump(history.history, open(checkpoint_path + ".losses.pickle", "wb"), pickle.HIGHEST_PROTOCOL)

        if do_eval:
            min_loss = np.min(history.history[loss_name])
            return model, min_loss
        else:
            return model, history

    @staticmethod
    def train_dnn_w_auxiliary(model, train_set_aux, train_set_main, validation_set, num_epochs, steps_per_epoch,
                              validation_steps, early_stopping_patience=12, with_checkpoints=True, do_eval=True,
                              do_early_stopping=True, checkpoint_path="./tmp/model.h5", save_best_only=True,
                              do_burn_in=True, auxiliary_early_stopping_patience=12, loss_name="val_loss"):
        show_output = 2
        callbacks = []

        if do_early_stopping:
            callbacks.append(EarlyStopping(monitor=loss_name,
                                           patience=early_stopping_patience))

        if with_checkpoints:
            callbacks.append(NoHDF5ModelCheckpoint(filepath=checkpoint_path,
                                                   monitor=loss_name,
                                                   save_best_only=save_best_only,
                                                   save_weights_only=False,
                                                   verbose=0))

        _, history = ModelTrainer.fit_generator(model,
                                                train_set_aux=train_set_aux, train_set_main=train_set_main,
                                                epochs=num_epochs, steps_per_epoch=steps_per_epoch,
                                                validation_data=validation_set, validation_steps=validation_steps,
                                                outer_callbacks=callbacks, verbose=show_output,
                                                do_burn_in=do_burn_in,
                                                auxiliary_early_stopping_patience=auxiliary_early_stopping_patience)

        if with_checkpoints:
            # Save model loss history.
            pickle.dump(history.history, open(checkpoint_path + ".losses.pickle", "wb"), pickle.HIGHEST_PROTOCOL)

        if do_eval:
            min_loss = np.min(history.history[loss_name])
            return model, min_loss
        else:
            return model, history

    @staticmethod
    def fit_generator(model, train_set_aux, train_set_main, epochs, steps_per_epoch,
                      verbose, validation_data, validation_steps, outer_callbacks,
                      do_burn_in=True,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      auxiliary_early_stopping_patience=12):
        model, aux_model = model

        validation_data_main = main_generator_wrapper(validation_data, validation_steps)
        validation_data_aux = main_generator_wrapper(validation_data, validation_steps, is_aux=True)

        # Prepare display labels.
        out_labels = model.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (outer_callbacks or []) + [history]
        if verbose:
            callbacks += [cbks.ProgbarLogger(count_mode='steps')]
        callbacks = cbks.CallbackList(callbacks)

        callbacks.set_model(model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': True,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        # prepare aux_callbacks
        aux_history = cbks.History()
        aux_early_stopping = EarlyStopping(patience=auxiliary_early_stopping_patience)
        aux_callbacks = [cbks.BaseLogger()] + [aux_history] + [aux_early_stopping]
        if verbose:
            aux_callbacks += [cbks.ProgbarLogger(count_mode='steps')]
        aux_callbacks = cbks.CallbackList(aux_callbacks)

        aux_callbacks.set_model(aux_model)
        aux_callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': True,
            'metrics': callback_metrics,
        })
        aux_callbacks.on_train_begin()

        def get_outputs(generator):
            outputs = next(generator)
            sample_weights, sample_weights_aux = None, None
            if len(outputs) == 3:
                x, (y, y_aux), sample_weights = outputs
                sample_weights, sample_weights_aux = sample_weights
            else:
                x, (y, y_aux) = outputs
            return x, y, y_aux, sample_weights, sample_weights_aux

        enqueuers = []

        def make_enqueuer_generator(gen):
            enqueuer = GeneratorEnqueuer(gen,
                                         use_multiprocessing=use_multiprocessing,
                                         wait_time=0.01)  # In seconds.
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            enqueuers.append(enqueuer)

            return output_generator

        try:
            aux_output_generator = make_enqueuer_generator(train_set_aux)
            main_output_generator = make_enqueuer_generator(train_set_main)

            aux_epoch = 0
            model.stop_training = False
            aux_model.stop_training = False
            for epoch in range(epochs):
                if model.stop_training:
                    break

                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0
                aux_batch_index = 0

                num_aux_epochs = 1
                if do_burn_in and epoch == 0:
                    num_aux_epochs = 5
                    print("INFO: Performing burn-in phase with", num_aux_epochs, "iterations on the auxiliary tasks.",
                          file=sys.stderr)

                for aux_batch_idx in range(num_aux_epochs):
                    if aux_model.stop_training:
                        break

                    aux_callbacks.on_epoch_begin(aux_epoch)

                    for iteration in range(steps_per_epoch):
                        x_aux, _, y_aux, _, sample_weights_aux = get_outputs(aux_output_generator)

                        if isinstance(x_aux, list):
                            batch_size = x_aux[0].shape[0]
                        elif isinstance(x_aux, dict):
                            batch_size = list(x_aux.values())[0].shape[0]
                        else:
                            batch_size = x_aux.shape[0]

                        aux_batch_logs = {"batch": aux_batch_index, "size": batch_size}
                        aux_callbacks.on_batch_begin(aux_batch_index, aux_batch_logs)

                        aux_outs = aux_model.train_on_batch(x_aux, y_aux,
                                                            sample_weight=sample_weights_aux)

                        if not isinstance(aux_outs, list):
                            aux_outs = [aux_outs]

                        for l, o in zip(out_labels, aux_outs):
                            aux_batch_logs[l] = o

                        aux_callbacks.on_batch_end(aux_batch_index, aux_batch_logs)
                        aux_batch_index += 1

                    aux_val_outs = aux_model.evaluate_generator(validation_data_aux,
                                                                validation_steps)

                    if not isinstance(aux_val_outs, list):
                        aux_val_outs = [aux_val_outs]

                    aux_epoch_logs = {}

                    # Same labels assumed.
                    for l, o in zip(out_labels, aux_val_outs):
                        aux_epoch_logs['val_' + l] = o

                    aux_callbacks.on_epoch_end(aux_epoch, aux_epoch_logs)
                    aux_epoch += 1

                for iteration in range(steps_per_epoch):
                    x, y, _, sample_weights, _ = get_outputs(main_output_generator)

                    # build batch logs
                    batch_logs = {}
                    if isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]

                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size

                    callbacks.on_batch_begin(batch_index, batch_logs)

                    outs = model.train_on_batch(x, y, sample_weight=sample_weights)

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)

                    # Construct epoch logs.
                    epoch_logs = {}
                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if steps_done >= steps_per_epoch:
                        val_outs = model.evaluate_generator(validation_data_main,
                                                            validation_steps)

                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]

                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

                        callbacks.on_epoch_end(epoch, epoch_logs)
                        epoch += 1

                        if model.stop_training:
                            break

        finally:
            for enqueuer in enqueuers:
                enqueuer.stop()

        callbacks.on_train_end()
        aux_callbacks.on_train_end()

        return model, history
