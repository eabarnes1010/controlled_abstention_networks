"""Abstention loss function classes and associated alpha updater classes for regression.

"""
import sys
from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "March 21 2021"

# =============================================================================
# LOSS FUNCTIONS CLASSES
# =============================================================================
    
# -----------------------------------------------------------------------------
class AbstentionLogLoss(tf.keras.losses.Loss):
    """Regression log-loss with an abstention option based on
       tau. This formulation also uses a PID or constant alpha as
       indicated by the updater.

    """
    def __init__(self, tau_coarse, tau_fine, updater, spinup_epochs,coarse_epochs):
        super().__init__()
        self.tau_coarse = tf.constant(tau_coarse, dtype=tf.float64)
        self.tau_fine = tf.constant(tau_fine, dtype=tf.float64)
        self.tau = tf.Variable(tau_coarse, trainable=False, dtype=tf.float64)
        
        self.updater = updater
        self.spinup_epochs = tf.constant(spinup_epochs, dtype=tf.int32)
        self.coarse_epochs = tf.constant(coarse_epochs, dtype=tf.int32)

    def __str__(self):
        return (
            f"AbstentionLogLoss(updater={self.updater.__str__()})"
        )

    def call(self, y_true, y_pred):
        mean = tf.cast(y_pred[:,0], tf.float64)
        std = tf.clip_by_value(tf.cast(y_pred[:,1], tf.float64),
                               clip_value_min=1.0e-10,
                               clip_value_max=1.0e10)        
        norm_dist = tfp.distributions.Normal(mean,std)
        
        p = tf.cast(norm_dist.prob(tf.cast(y_true[:,0],tf.float64)), tf.float64)    # likehood of y_true    
        p = tf.clip_by_value(p,clip_value_min=1.0e-10,clip_value_max=1.0e10)

        # set TAU
        if self.updater.current_epoch < self.spinup_epochs+self.coarse_epochs:
            self.tau.assign(self.tau_coarse)
        else:
            self.tau.assign(self.tau_fine)
                
        # determine loss function for spinup and abstention training
        if(self.updater.current_epoch < self.spinup_epochs):
            loss = -tf.math.log(p)                                                  # standard baseline RegressLogLoss
        else:                
            beta = tf.math.square(tf.math.divide(self.tau_coarse,std))              # factor of tau**2/sigma**2
            confidence = tf.math.minimum(tf.cast(1.,tf.float64),beta)               # ensuring samples with sigma<tau are all equally weighted
            loss = tf.math.multiply(-tf.math.log(p),confidence) - self.updater.alpha*tf.math.log(confidence)

            
        self.updater.update(y_pred, self.tau)
        return tf.reduce_mean(loss, axis=-1)
    
# -----------------------------------------------------------------------------
class RegressLogLoss(tf.keras.losses.Loss):
    """Abstention loss function for regression which 
       penalized abstentions.
    """
    def __init__(self,):
        super().__init__()

    def __str__(self):
        return (
            f"RegressBaseline()"
        )

    def call(self, y_true, y_pred):
        mean = tf.cast(y_pred[:,0], tf.float64)
        std = tf.clip_by_value(tf.cast(y_pred[:,1], tf.float64),
                               clip_value_min=1.0e-10,
                               clip_value_max=1.0e10)        
        norm_dist = tfp.distributions.Normal(mean,std)
        
        p = tf.cast(norm_dist.prob(tf.cast(y_true[:,0],tf.float64)), tf.float64)    # likehood of y_true    
        p = tf.clip_by_value(p,clip_value_min=1.0e-10,clip_value_max=1.0e10)
        
        loss = -tf.math.log(p)
        return tf.reduce_mean(loss, axis=-1)   
    
# -----------------------------------------------------------------------------
class StandardMSE(tf.keras.losses.Loss):
    """Standard mean squared error for our 2-output regression setup.
    """
    def __init__(self,):
        super().__init__()

    def __str__(self):
        return (
            f"StandardMSE()"
        )

    def call(self, y_true, y_pred):
        
        error = tf.cast(y_pred[:,0] - y_true[:,0],tf.float64)
        loss = tf.math.square(error)
        return tf.reduce_mean(loss, axis=-1)     
    
# -----------------------------------------------------------------------------
class StandardMAE(tf.keras.losses.Loss):
    """Standard mean absolute error for our 2-output regression setup.
    """
    def __init__(self,):
        super().__init__()

    def __str__(self):
        return (
            f"StandardMSE()"
        )

    def call(self, y_true, y_pred):
        
        error = tf.cast(y_pred[:,0] - y_true[:,0],tf.float64)
        loss = tf.math.abs(error)
        return tf.reduce_mean(loss, axis=-1)       


# =============================================================================
# ALPHA UPDATER CLASSES
# =============================================================================

# -----------------------------------------------------------------------------
class AlphaUpdater(ABC):
    """The abstract base class for all alpha updaters.

    Attributes
    ----------
    setpoint : float
        The abstention setpoint.

    alpha : float
        The current alpha value.

    Requirements
    ------------
    * An instance of an AlphaUpdaterCallBack must be included in the callbacks
    of the model.fit.

    """
    current_epoch = tf.Variable(0, trainable=False)
    current_batch = tf.Variable(0, trainable=False)

    def __init__(self, setpoint, alpha_init):
        """The abstract base class for all alpha update classes.

        Arguments
        ---------
        setpoint : float
            The abstention setpoint.

        alpha_init : float
            The initial alpha.

        """
        self.setpoint = tf.constant(setpoint, dtype=tf.float64)
        self.alpha_init = tf.constant(alpha_init, dtype=tf.float64)

        self.alpha = tf.Variable(alpha_init, trainable=False, dtype=tf.float64)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, y_pred):
        raise NotImplementedError()
        
# -----------------------------------------------------------------------------
class Constant(AlphaUpdater):
    """Discrete-time constant updater using moving averages of batch counts.

    The penalty factor, alpha, is held fixed.

    """
    def __init__(
        self,
        setpoint,
        alpha_init,
    ):
        super().__init__(setpoint, alpha_init)

    def __str__(self):
        return(
            f"Colorado("
            + f"setpoint={self.setpoint}, "
            + f"alpha_init={self.alpha_init}, "
        )

    def update(self, y_pred, tau):
        self.alpha.assign(tf.cast(self.alpha_init,tf.float64))

class Colorado(AlphaUpdater):
    """Discrete-time PID updater using moving averages of batch counts.

    The penalty factor, alpha, is updated every batch, based on the abstention
    fraction of the previous <length> batches combined.

    The controller is an implementation of a discrete-time PID controller
    (velocity algorithm). See, for example, Visioli (2006) Equation (1.38).

    Attributes
    ----------
    setpoint : float
        The abstention fraction setpoint.

    alpha_init : float
        The initial alpha.

    k : array-like, shape=(3,), dtype=float
        The three tuning parameters; e.g. k=(0.50, 0.40, 0.10).

    clip_at : array-like, shape=(2,), dtype=float
        The lower and upper clip points for defining the maximum change in
        alpha on any update; e.g. clip_at=(-0.5, 0.5).

    length : int
        The length of the cyclic queue; e.g. length=50.

    Requirements
    ------------
    * 0 < setpoint < 1
    * 0 <= alpha_init
    * k[1] - 2*k[2] > 0
    * k[0] - k[1] + k[2] > 0
    * k[2] > 0
    * clip_at[0] < 0 < clip_at[1]
    * length >= 1

    """
    def __init__(
        self,
        setpoint,
        alpha_init,
        length=10,        
        k=(0.50, 0.25, 0.05),       # These need to be tuned for each data set
        clip_at=(-1.0, 1.0),
    ):
        super().__init__(setpoint, alpha_init)

        self.k = tf.constant(k, dtype=tf.float64)
        self.clip_at = tf.constant(clip_at, dtype=tf.float64)
        self.length = tf.constant(length, dtype=tf.int32)

        self.batch_votes       = CyclicQueue(length)
        self.batch_abstentions = CyclicQueue(length)

        self.previous_error    = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self.penultimate_error = tf.Variable(0.0, trainable=False, dtype=tf.float64)

    def __str__(self):
        return(
            f"Colorado("
            + f"setpoint={self.setpoint}, "
            + f"alpha_init={self.alpha_init}, "
            + f"k={self.k}, "
            + f"clip_at={self.clip_at}, "
            + f"length={self.length})"
        )

    def update(self, y_pred, tau):
        
        abstentions = tf.math.greater_equal(tf.cast(y_pred[:,1],tf.float64),tau)
        votes = tf.math.less(tf.cast(y_pred[:,1],tf.float64),self.alpha)
        
        self.batch_votes.put(tf.math.count_nonzero(votes, dtype=tf.int32))
        self.batch_abstentions.put(tf.math.count_nonzero(abstentions, dtype=tf.int32))

        total_votes = self.batch_votes.sum()
        total_abstentions = self.batch_abstentions.sum()

        abstention_fraction = tf.math.divide(total_abstentions, total_votes + total_abstentions)
        error = self.setpoint - tf.cast(abstention_fraction, tf.float64)
#         tf.print(self.setpoint, output_stream=sys.stderr)
        
        adjustment = tf.clip_by_value(
            self.k[0] * error
            - self.k[1] * self.previous_error
            + self.k[2] * self.penultimate_error,
            self.clip_at[0], self.clip_at[1]
        )
        self.alpha.assign(tf.math.maximum(tf.cast(0.0, tf.float64), self.alpha - adjustment))

        self.penultimate_error.assign(self.previous_error)
        self.previous_error.assign(error)

        
# =============================================================================
# CYCLIC QUEUE
# =============================================================================

class CyclicQueue:
    """A tensorflow-compatable, cyclic queue.

    Attributes
    ----------
    length : int
        The length of the cyclic queue.

    queue : array-like, shape=(length,), dtype=int32
        The storage for the queued values.

    index : int
        The index of the current top of the cyclic queue.

    """
    def __init__(self, length):
        self.length = tf.constant(length, dtype=tf.int32)
        self.queue = tf.Variable(tf.zeros([length], dtype=tf.int32), dtype=tf.int32)
        self.index = tf.Variable(0, dtype=tf.int32)

    def put(self, value):
        self.queue[self.index].assign(value)
        self.index.assign((self.index + 1) % self.length)

    def sum(self):
        return tf.math.reduce_sum(self.queue)


# =============================================================================
# ALPHA UPDATER CALLBACK
#
#   An instance of an AlphaUpdaterCallBack MUST BE included in the callbacks
#   of the model.fit.
# =============================================================================

# -----------------------------------------------------------------------------
class AlphaUpdaterCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(AlphaUpdater.current_epoch, epoch)

    def on_batch_begin(self, batch, logs=None):
        tf.keras.backend.set_value(AlphaUpdater.current_batch, batch)
