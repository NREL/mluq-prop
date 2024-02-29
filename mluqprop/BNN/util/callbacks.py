#!/usr/bin/env python3

"""
Bayesian Neural Network Callbacks
"""

import os

import numpy as np
import tensorflow as tf
from mluqprop.BNN.util.metrics import (
    compute_snr,
    compute_snr_flipout,
    vec_moment_distance,
    wass1d,
)

__author__ = "Graham Pash"


#############################
# Callbacks
#############################
# ========================================================================
# learning rate scheduler
class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        final_learning_rate,
        epochs,
        steps_per_epoch,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.final_learning_rate = final_learning_rate
        self._factor = np.log(
            self.final_learning_rate / self.initial_learning_rate
        ) / (self.epochs / 2)

    def __call__(self, step):
        if step < (self.steps_per_epoch * self.epochs) // 2:
            return self.initial_learning_rate
        else:
            return max(
                self.initial_learning_rate * tf.math.exp(self._factor),
                self.final_learning_rate,
            )


# ========================================================================
# Metrics for evaluation and monitoring.
class Metrics(tf.keras.callbacks.Callback):
    """Custom callback for monitoring Wasserstein distance, SNR, and moment distance."""

    def __init__(self, uips, xval, yval, simparams, verbose: bool = False):
        """Custom

        Args:
            uips: Inducing points.
            xval: Validation inputs.
            yval: Validation outputs.
            simparams: Simulation parameters namespace from input deck.
            verbose (bool, optional): Print metrics at each epoch. Defaults to False.
        """
        self.uips = uips
        self.xval = xval
        self.yval = yval
        self.nalea = simparams.nalea
        self.nepi = simparams.nepi
        self.skip = simparams.skip
        self.verbose = verbose
        if simparams.model_type == "flipout":
            self.flipout = True
        else:
            self.flipout = False
        self.checkpath = simparams.checkpath

    def on_train_begin(self, logs={}):
        self.wasserstein = []
        self.medsnr = []
        self.meandist = []
        self.stddist = []
        self.epochs = []
        self.uips_meannorm = np.linalg.norm(self.uips["mean"])
        self.uips_stdnorm = np.linalg.norm(self.uips["aleat"])

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.skip == 0:
            # report after first epoch and at desired frequency

            import time

            self.epochs.append(epoch)

            start = time.time()
            # self.wasserstein.append(wass1d(self.yval, self.model.predict(self.xval)))
            # model(X).sample() appears to be faster than model.predict(X)
            self.wasserstein.append(
                wass1d(self.yval, self.model(self.xval).sample().numpy())
            )
            print(
                f"Time to compute Wasserstein distance: {time.time() - start:.2e} seconds"
            )

            start = time.time()
            if self.flipout:
                self.medsnr.append(np.median(compute_snr_flipout(self.model)))
            else:
                self.medsnr.append(np.median(compute_snr(self.model)))
            print(f"Time to compute SNR: {time.time() - start:.2e} seconds")

            start = time.time()
            md, sd = vec_moment_distance(
                self.model, uips=self.uips, nalea=self.nalea, nepi=self.nepi
            )
            print(
                f"Time to compute moment distance: {time.time() - start:.2e} seconds"
            )
            self.meandist.append(md / self.uips_meannorm)
            self.stddist.append(sd / self.uips_stdnorm)

            if self.verbose:
                # optional printing to console
                print(
                    f"Wasserstein distance at epoch {epoch}: {self.wasserstein[-1]:.3e}"
                )
                print(f"Median SNR at epoch {epoch}: {self.medsnr[-1]:.3e}")
                print(
                    f"Relative norm of first moment with inducing points {epoch}: {self.meandist[-1]:.3e}"
                )
                print(
                    f"Relative norm of second moment with inducing points {epoch}: {self.stddist[-1]}"
                )

            # append to log file
            if epoch == 0:
                with open(
                    os.path.join(self.checkpath, "metrics.log"), "w"
                ) as f:
                    f.write("epoch,wasserstein,medsnr,meandist,stddist\n")
                    f.write(
                        f"{epoch},{self.wasserstein[-1]},{self.medsnr[-1]},{self.meandist[-1]},{self.stddist[-1]}\n"
                    )
            else:
                with open(
                    os.path.join(self.checkpath, "metrics.log"), "a"
                ) as f:
                    f.write(
                        f"{epoch},{self.wasserstein[-1]},{self.medsnr[-1]},{self.meandist[-1]},{self.stddist[-1]}\n"
                    )

        return


class BNNCallbacks:
    """Class implementing common callbacks for BNN training."""

    def __init__(self, simparams, Xmetric, Ymetric, N, verbose):
        """
        Args:
            simparams: Simulation parameters object.
            Xmetric: Training data inputs for metrics callback.
            Ymetric: Training data outputs for metrics callback.
            N: Number of training data points for metrics callback.
            verbose: Verbosity flag.
        """
        self.simparams = simparams
        self.Xmetric = Xmetric
        self.Ymetric = Ymetric
        self.verbose = verbose
        self.N = N

    # Return the best model weights path.
    def best_ckpt_path(self):
        return os.path.join(self.simparams.checkpath, "best/", "best")

    # Custom metrics logging.
    def metrics_cb(self):
        # Load in the inducing points
        uips = np.load(self.simparams.uips_fpath)
        return Metrics(
            uips,
            self.Xmetric,
            self.Ymetric,
            self.simparams,
            verbose=self.verbose,
        )

    # Early stopping if model training plateaus.
    def earlystop_cb(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.simparams.patience,
            restore_best_weights=True,
        )

    # Custom learning rate scheduler.
    def lrschedule_cb(self):
        return tf.keras.callbacks.LearningRateScheduler(
            LRSchedule(
                initial_learning_rate=self.simparams.learning_rate,
                final_learning_rate=self.simparams.final_lr,
                epochs=self.simparams.epochs,
                steps_per_epoch=self.N / self.simparams.batch_size,
            )
        )

    # Reduce learning rate on plateau.
    def reduce_lr_cb(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=self.simparams.patience,
            mode="min",
        )

    # Checkpoint the best model.
    def best_ckpt_cb(self, ckpt_path=None):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.best_ckpt_path() if ckpt_path is None else ckpt_path,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
        )

    # Checkpoint ever so many epochs.
    def epoch_ckpt_cb(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                self.simparams.checkpath, "epoch/", "epoch-{epoch:05d}.h5"
            ),
            save_weights_only=True,
            monitor="loss",
            mode="min",
            period=self.simparams.skip,
        )
