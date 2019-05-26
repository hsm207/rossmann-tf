# Implement One Cycle Policy Algorithm in the Keras Callback Class
# Adapted from https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from numbers import Number
from typing import Callable, Tuple, List
import numpy as np
import sys
import itertools
import matplotlib.pyplot as plt

AnnealFunc = Callable[[Number, Number, float], Number]


def annealing_exp(start: float, end: float, pct: float) -> float:
    return start * (end / start) ** pct


def annealing_cos(start: float, end: float, pct: float) -> float:
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class Stepper:
    def __init__(self, start: Number, end: Number, steps: int, anneal_func: AnnealFunc):
        self.n = 0
        self.total_steps = steps
        self.start = start
        self.end = end
        self.anneal_func = anneal_func

    def step(self) -> Number:
        self.n = self.n + 1
        progress = self.n / self.total_steps
        next_val = self.anneal_func(start=self.start,
                                    end=self.end,
                                    pct=progress)

        return next_val

    @property
    def is_done(self) -> bool:
        return self.n >= self.total_steps


class OneCycleScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 lr_max: float,
                 num_epochs: int,
                 num_observations: int,
                 batch_size: int,
                 moms: Tuple[float, float] = (0.95, 0.85),
                 div_factor: float = 25.,
                 pct_start: float = 0.3
                 ):

        start_lr = lr_max / div_factor
        end_lr = start_lr / 1e4
        mom_start, mom_end = moms

        n_iters = num_observations // batch_size * num_epochs
        # number of steps for the first phase of the cycle
        phase1_iter = int(n_iters * pct_start)
        # number of steps for the second phase of the cycle
        phase2_iter = n_iters - phase1_iter
        # attribute to hold the params of the cycle
        self.phases = ((phase1_iter, annealing_cos),
                       (phase2_iter, annealing_cos))

        self.lr_schedule = self._build_schedule((start_lr, lr_max),
                                                (lr_max, end_lr))
        self.mom_schedule = self._build_schedule((mom_start, mom_end),
                                                 (mom_end, mom_start))
        # index to control phase 1 or phase 2 part of the schedule
        self.schedule_idx = 0

        self.lr_hist = []
        self.mom_hist = []

    def _build_schedule(self, *schedule_rates: Tuple[float, float]):
        return [
            Stepper(start=start_rate, end=end_rate, steps=steps, anneal_func=func)
            for ((start_rate, end_rate), (steps, func)) in zip(schedule_rates, self.phases)
        ]

    def on_train_begin(self, logs=None):
        self.opt = self.model.optimizer

        lr = self.lr_schedule[0].start
        mom = self.mom_schedule[0].start

        K.set_value(self.opt.learning_rate, lr)
        K.set_value(self.opt.beta_1, mom)

        self.lr_hist.append(lr)
        self.mom_hist.append(mom)

    def on_train_batch_end(self, batch, logs=None):
        i = self.schedule_idx
        lr = self.lr_schedule[i].step()
        mom = self.mom_schedule[i].step()

        self.lr_hist.append(lr)
        self.mom_hist.append(mom)

        K.set_value(self.opt.learning_rate, lr)
        K.set_value(self.opt.beta_1, mom)

        if self.lr_schedule[i].is_done:
            self.schedule_idx += 1

        # stop training once both phases are done
        if self.schedule_idx >= len(self.lr_schedule):
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        logs['lr_history'] = self.lr_hist
        logs['mom_history'] = self.mom_hist

        self.lr_hist = []
        self.mom_hist = []


class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, start_lr: float = 1e-7,
                 end_lr: float = 10,
                 num_iter: int = 100,
                 stop_div: bool = True,
                 anneal_func: AnnealFunc = annealing_cos):
        super(LRFinder, self).__init__()
        self.stop_div = stop_div
        self.lr_schedule = Stepper(start=start_lr,
                                   end=end_lr,
                                   steps=num_iter,
                                   anneal_func=anneal_func)
        self.history = {}
        self.loss_history = []
        self.lr_history = []

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.lr_schedule.step()
        n = self.lr_schedule.n

        self.lr_history.append(lr)

        K.set_value(self.model.optimizer.lr, lr)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs['loss']
        self.loss_history.append(loss)

        if self.lr_schedule.is_done:
            self.model.stop_training = True

    def on_epoch_end(self, batch, logs=None):
        logs['loss_history'] = self.loss_history
        logs['lr_history'] = self.lr_history

        self.loss_history = []
        self.lr_history = []


def fit_one_cycle(model: tf.keras.Model,
                  train_ds: tf.data.Dataset,
                  val_ds: tf.data.Dataset,
                  lr_max: float,
                  num_epochs: int,
                  num_train_observations: int,
                  batch_size: int,
                  moms: Tuple[float, float] = (0.95, 0.85),
                  div_factor: float = 25.,
                  pct_start: float = 0.3,
                  callbacks: List[tf.keras.callbacks.Callback] = []
                  ):
    """
    Train model using one cycle training as outlined in
    Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf) and
    [implemented](https://github.com/fastai/fastai/blob/953214c6b78a0ec96a79d57e56aaf8d9900033de/fastai/train.py#L14)
    in the [fastai library

    :param model: a compiled keras model to train
    :param train_ds: the training set
    :param val_ds: the validation set
    :param lr_max: maximum learning identified using find_lr
    :param num_epochs: number of epochs to do training
    :param num_train_observations: number of observations in train_ds
    :param batch_size: batch size for training
    :param moms: the schedule for the optimizer's momentum
    :param div_factor: factor used to compute the starting learning rate
    :param pct_start: fraction that the iterations to designated as phase 1 of the one cycle training
    :param callbacks: Other callbacks to be past to model
    :return:
    """

    one_cycle_callback = OneCycleScheduler(lr_max=lr_max,
                                           num_epochs=num_epochs,
                                           num_observations=num_train_observations,
                                           batch_size=batch_size,
                                           moms=moms,
                                           div_factor=div_factor,
                                           pct_start=pct_start)

    callbacks.append(one_cycle_callback)

    results = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=num_epochs,
                        callbacks=callbacks,
                        verbose=1)

    lr_history = results.history['lr_history']
    mom_history = results.history['mom_history']
    train_loss = results.history['loss']
    val_loss = results.history['val_loss']

    lr_history = list(itertools.chain(*lr_history))
    mom_history = list(itertools.chain(*mom_history))

    n_epochs = list(range(1, len(val_loss) + 1))
    n_steps = list(range(1, len(lr_history) + 1))

    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(w=10, h=10)

    # TODO: Check if train loss is computed over all batches at the end of epoch
    axes[0].plot(n_epochs, train_loss[:len(val_loss)])
    axes[0].plot(n_epochs, val_loss)
    axes[0].legend(['Train', 'Validation'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(n_steps, lr_history)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Learning Rate')

    axes[2].plot(n_steps, mom_history)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Momentum')

    return fig


def find_lr(model, ds, start_lr: float = 1e-7,
            end_lr: float = 10,
            num_iter: int = 100,
            stop_div: bool = True,
            anneal_func: AnnealFunc = annealing_exp):
    """
    Produce a plot to identify the maximum learning rate for one cycle training
    :param model: A compiled Keras model
    :param ds: A tf.dataset
    :param start_lr: Minimum learning rate to begin search
    :param end_lr: Maximum learning rate to end search at
    :param num_iter: Number of steps to reach end_lr from start_lr
    :param stop_div: Stop when loss starts to diverge (not implemented yet)
    :param anneal_func: The annealing function to apply on the learning rate
    :return: A matplotlib figure
    """
    lr_finder_callback = LRFinder(start_lr=start_lr,
                                  end_lr=end_lr,
                                  num_iter=num_iter,
                                  anneal_func=anneal_func
                                  )

    # set to a really high epoch because we assume num_iter is a lot smaller
    results = model.fit(ds,
                        epochs=sys.maxsize,
                        callbacks=[lr_finder_callback],
                        verbose=1)

    results.history.keys()
    lr, loss = results.history['lr_history'], results.history['loss_history']
    lr = list(itertools.chain(*lr))
    loss = list(itertools.chain(*loss))
    i = list(range(1, len(lr) + 1))

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(h=8, w=10)

    axes[0].plot(lr, loss)
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_xscale('log')
    axes[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    axes[1].plot(i, lr)
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_xlabel('Step')

    return fig
