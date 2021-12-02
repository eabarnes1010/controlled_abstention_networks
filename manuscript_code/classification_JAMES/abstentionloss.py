"""Abstention loss function classes and associated alpha updater classes.

loss function classes
---------------------
* DACLoss(tf.keras.losses.Loss):
    Modified k-class cross-entropy per-sample loss with spinup, as defined by
    Sunil Thulasidasan.

* NotWrongLoss(tf.keras.losses.Loss):
    Abstention loss function that highlights being "not wrong", rather than
    being right, while penalizing abstentions.

alpha updater classes
---------------------
* Colorado(AlphaUpdater):
    Discrete-time PID updater using moving averages of batch counts.

* Minnesota(AlphaUpdater):
    Discrete-time PID updater using non-overlapping sets of batch counts.

* Washington(AlphaUpdater):
    A PID-like updater based on the code of Sunil Thulasidasan.

Notes
-----
* All abstention loss functions have a penalty term to control the fraction of
abstentions. The penalty term takes the form:

    penalty = -alpha * log(likelihood of not abstaining)

* The updater determines how alpha changes throughout the training. The changes
can be systematic, or using control theory to achieve a specified setpoint.

"""
import sys
from abc import ABC, abstractmethod
import tensorflow as tf

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "23 January 2021"

# =============================================================================
# DAC ABSTENTION LOSS FUNCTIONS CLASSES
# =============================================================================


# -----------------------------------------------------------------------------
class DACLoss(tf.keras.losses.Loss):
    """Modified k-class cross-entropy per-sample loss with spinup.

    The standard cross-entropy training loss is computed during the initial
    spinup epochs, then the modified k-class cross-entropy per-sample loss
    is computed thereafter.

    The modified k-class cross-entropy per-sample loss is a direct application
    of Thulasidasan, et al. (2019), Equation (1), which is the same as
    Thulasidasan (2020), Equation (3.1).

    Attributes
    ----------
    updater : an instance of an AlphaUpdater.

    spinup_epochs : int
        The number of initial spinup epochs.

    Requirements
    ------------
    * The abstention category must be the last category.
    * The y_pred must be the output from a softmax, so that y_pred sums to 1.

    References
    ----------
    * Thulasidasan, S., T. Bhattacharya, J. Bilmes, G. Chennupati, and
        J. Mohd-Yusof, 2019, Combating Label Noise in Deep Learning Using
        Abstention. arXiv [stat.ML].

    * Thulasidasan, S., 2020, Deep Learning with Abstention: Algorithms for
        Robust Training and Predictive Uncertainty, PhD Dissertation,
        University of Washington.
        https://digital.lib.washington.edu/researchworks/handle/1773/45781

    """
    def __init__(self, updater, spinup_epochs):
        super().__init__()
        self.updater = updater
        self.spinup_epochs = tf.constant(spinup_epochs, dtype=tf.int32)

    def __str__(self):
        return(
            f"DACLoss(updater={self.updater.__str__()}, spinup_epochs={self.spinup_epochs})"
        )

    def call(self, y_true, y_pred):
        q = tf.cast(1 - y_pred[:, -1], tf.float64)                  # likelihood of NOT abstaining
        logq = tf.math.log(q)

        r = tf.cast(tf.boolean_mask(y_pred, y_true), tf.float64)    # likelihood of being correct
        logr = tf.math.log(r)

        if self.updater.current_epoch < self.spinup_epochs:
            loss = logq - logr
        else:
            loss = (q - self.updater.alpha)*logq - q*logr

        self.updater.update(y_pred)
        return tf.reduce_mean(loss, axis=-1)


# -----------------------------------------------------------------------------
class NotWrongLoss(tf.keras.losses.Loss):
    """Abstention loss function that highlights being "not wrong", rather than
     being right, while penalizing abstentions.

    Attributes
    ----------
    updater : an instance of an AlphaUpdater.

    Requirements
    ------------
    * The abstention category must be the last category.
    * The y_pred must be the output from a softmax, so that y_pred sums to 1.

    """
    def __init__(self, updater, spinup_epochs):
        super().__init__()
        self.updater = updater
        self.spinup_epochs = tf.constant(spinup_epochs, dtype=tf.int32)

    def __str__(self):
        return (
            f"NotWrongLoss(updater={self.updater.__str__()})"
        )

    def call(self, y_true, y_pred):
        q = tf.cast(1 - y_pred[:, -1], tf.float64)                  # likelihood of NOT abstaining
        r = tf.cast(tf.boolean_mask(y_pred, y_true), tf.float64)    # likelihood of being correct
        s = tf.cast(y_pred[:, -1], tf.float64)                      # likelihood of abstaining

        if self.updater.current_epoch < self.spinup_epochs:
            loss = tf.math.log(q) - tf.math.log(r)
        else:
            loss = -tf.math.log(r+s) - self.updater.alpha * tf.math.log(q)

        self.updater.update(y_pred)
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
class Minnesota(AlphaUpdater):
    """Discrete-time PID updater using non-overlapping sets of batch counts.

    The penalty factor, alpha, is updated once every <length> batches,
    based on the abstention fraction of the previous <length> batches
    combined.

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
        The number of batches between updates; e.g. length=50.

    Requirements
    ------------
    * 0 < setpoint < 1
    * 0 <= alpha_init
    * k[1] - 2*k[2] > 0
    * k[0] - k[1] + k[2] > 0
    * k[2] > 0
    * clip_at[0] < 0 < clip_at[1]
    * length >= 1

    Notes
    -----
    * We can relate the components of the three K tuning parameters to the
    standard PID parameters as

        [ 0  1 -2 ]   [ k[0] ]           [ 1      ]
        [ 1 -1  1 ] * [ k[1] ]  =  K_p * [ dt/T_i ]
        [ 0  0  1 ]   [ k[2] ]           [ T_d/dt ]

    or equivalently

              [ 1  1  1 ]   [ 1      ]     [ k[0] ]
        K_p * [ 1  0  2 ] * [ dt/T_i ]  =  [ k[1] ]
              [ 0  0  1 ]   [ T_d/dt ]     [ k[2] ]

    References
    ----------
    * Visioli, A, 2006, Practical PID Control, Springer, ISBN-13: 9781846285851.

    """
    def __init__(
        self,
        setpoint,
        alpha_init,
        k=(0.50, 0.25, 0.10),       # These need to be tuned for each data set
        clip_at=(-0.25, 0.25),
        length=50
    ):
        super().__init__(setpoint, alpha_init)

        self.k = tf.constant(k, dtype=tf.float64)
        self.clip_at = tf.constant(clip_at, dtype=tf.float64)
        self.length = tf.constant(length, dtype=tf.int32)

        self.total_votes       = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.total_abstentions = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.previous_error    = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self.penultimate_error = tf.Variable(0.0, trainable=False, dtype=tf.float64)

    def __str__(self):
        return(
            f"Minnesota("
            + f"setpoint={self.setpoint}, "
            + f"alpha_init={self.alpha_init}, "
            + f"k={self.k}, "
            + f"clip_at={self.clip_at}, "
            + f"length={self.length})"
        )

    def update(self, y_pred):
        abstain_category = y_pred.get_shape()[-1] - 1

        predicted_category = tf.math.argmax(y_pred, axis=-1, output_type=tf.int64)
        votes = tf.math.not_equal(predicted_category, abstain_category)
        abstentions = tf.math.equal(predicted_category, abstain_category)

        self.total_votes.assign_add(tf.math.count_nonzero(votes))
        self.total_abstentions.assign_add(tf.math.count_nonzero(abstentions))

        if self.current_batch % self.length == 0:
            abstention_fraction = tf.math.divide(self.total_abstentions, self.total_votes + self.total_abstentions)
            error = self.setpoint - tf.cast(abstention_fraction, tf.float64)

            adjustment = tf.clip_by_value(
                self.k[0] * error
                - self.k[1] * self.previous_error
                + self.k[2] * self.penultimate_error,
                self.clip_at[0], self.clip_at[1]
            )
            self.alpha.assign(tf.math.maximum(tf.cast(0.0, tf.float64), self.alpha - adjustment))

            self.penultimate_error.assign(self.previous_error)
            self.previous_error.assign(error)

            self.total_votes.assign(0)
            self.total_abstentions.assign(0)


# -----------------------------------------------------------------------------
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
        k=(0.50, 0.25, 0.05),       # These need to be tuned for each data set
        clip_at=(-0.05, 0.05),
        length=10
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

    def update(self, y_pred):
        abstain_category = y_pred.get_shape()[-1] - 1

        predicted_category = tf.math.argmax(y_pred, axis=-1, output_type=tf.int64)
        votes = tf.math.not_equal(predicted_category, abstain_category)
        abstentions = tf.math.equal(predicted_category, abstain_category)

        self.batch_votes.put(tf.math.count_nonzero(votes, dtype=tf.int32))
        self.batch_abstentions.put(tf.math.count_nonzero(abstentions, dtype=tf.int32))

        total_votes = self.batch_votes.sum()
        total_abstentions = self.batch_abstentions.sum()

        abstention_fraction = tf.math.divide(total_abstentions, total_votes + total_abstentions)
        error = self.setpoint - tf.cast(abstention_fraction, tf.float64)

        adjustment = tf.clip_by_value(
            self.k[0] * error
            - self.k[1] * self.previous_error
            + self.k[2] * self.penultimate_error,
            self.clip_at[0], self.clip_at[1]
        )
        self.alpha.assign(tf.math.maximum(tf.cast(0.0, tf.float64), self.alpha - adjustment))

        self.penultimate_error.assign(self.previous_error)
        self.previous_error.assign(error)


# -----------------------------------------------------------------------------
class Washington(AlphaUpdater):
    """A PID-like updater based on the code of Sunil Thulasidasan.

    References
    ----------
    * Thulasidasan, S., 2020, Deep Learning with Abstention: Algorithms for
        Robust Training and Predictive Uncertainty, PhD Dissertation,
        University of Washington.
        https://digital.lib.washington.edu/researchworks/handle/1773/45781

    * https://github.com/thulas/dac-label-noise/blob/master/dac_loss.py

    """
    def __init__(
        self,
        setpoint,
        alpha_init,
        kp=0.10,
        ki=0.10,
        kd=0.05,
        dt=0.01,
        mu=0.05,
        clip_at=(-1.0, 1.0),
        length=0 #dummy variable for compatibilitiy
    ):
        super().__init__(setpoint, alpha_init)

        self.kp = tf.constant(kp, dtype=tf.float32)  # PID proportional coefficient
        self.ki = tf.constant(ki, dtype=tf.float32)  # PID integral coefficient
        self.kd = tf.constant(kd, dtype=tf.float32)  # PID derivative coefficient
        self.dt = tf.constant(dt, dtype=tf.float32)  # PID pseudo time increment
        self.mu = tf.constant(mu, dtype=tf.float32)  # smoothing factor
        self.clip_at = tf.constant(clip_at, dtype=tf.float32)  # clipping limit for control

        self.smoothed = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.integral = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def __str__(self):
        return (
                f"Washington("
                + f"setpoint={self.setpoint}, "
                + f"alpha_init={self.alpha_init}, "
                + f"kp={self.kp}, "
                + f"ki={self.ki}, "
                + f"kd={self.kd}, "
                + f"dt={self.dt}, "
                + f"mu={self.mu}, "
                + f"clip_at={self.clip_at})"
        )

    def update(self, y_pred):
        abstain_category = y_pred.get_shape()[-1] - 1

        predicted_category = tf.math.argmax(y_pred, axis=-1, output_type=tf.int64)
        voted = tf.math.not_equal(predicted_category, abstain_category)

        pv = self.mu * self.smoothed + (1.0 - self.mu) * tf.math.zero_fraction(voted)
        delta_pv = pv - self.smoothed
        self.smoothed.assign(pv)
        error = self.setpoint - pv

        proportional = self.kp * error
        self.integral.assign_add(tf.clip_by_value(self.ki * error * self.dt, self.clip_at[0], self.clip_at[1]))
        derivative = -self.kd * delta_pv / self.dt          # the -1 is from thesis!

        control = tf.clip_by_value(proportional + self.integral + derivative, self.clip_at[0], self.clip_at[1])
        self.alpha.assign(tf.math.maximum(0.0, self.alpha - control))


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
