# Implement One Cycle Policy Algorithm in the Keras Callback Class
# Adapted from https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from numbers import Number
from typing import Callable
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


class CyclicLR(tf.keras.callbacks.Callback):

    def __init__(self, base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum):

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.base_m = base_m
        self.max_m = max_m
        self.cyclical_momentum = cyclical_momentum
        self.step_size = step_size

        self.clr_iterations = 0.
        self.cm_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):

        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))

        if cycle == 2:
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.base_lr - (self.base_lr - self.base_lr / 100) * np.maximum(0, (1 - x))

        else:
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

    def cm(self):

        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))

        if cycle == 2:

            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.max_m

        else:
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            return self.max_m - (self.max_m - self.base_m) * np.maximum(0, (1 - x))

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

        if self.cyclical_momentum == True:
            if self.clr_iterations == 0:
                K.set_value(self.model.optimizer.momentum, self.cm())
            else:
                K.set_value(self.model.optimizer.momentum, self.cm())

    def on_batch_begin(self, batch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        if self.cyclical_momentum == True:
            self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

        if self.cyclical_momentum == True:
            K.set_value(self.model.optimizer.momentum, self.cm())
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

    fig.savefig('./figures/viz.png')

    return fig
