"""Classification with abstention metric classes and functions."""

import numpy as np
import tensorflow as tf


__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "January 11, 2021"

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprectionWarning)

# ------------------------------------------------------------------------
# FUNCTIONS
#
#   The following metric functions are used for comparison purposes and
#   plotting.  These are not necessarily tensorflow compliant.
# ------------------------------------------------------------------------


def perfect_acc(abst_setpoint, gameboard):
    """Compute the prediction accuracy assuming perfect classification
    based on the gameboard labels (i.e. before the mislabeling of pixels),
    and assuming that abstentions are applied to mislabeled pixels only.

    """
    correct_fraction = (gameboard.ncell - gameboard.pr_mislabel * gameboard.nnoisy) / gameboard.ncell
    predicted_fraction = 1.0 - abst_setpoint
    return np.minimum(1.0, correct_fraction/predicted_fraction)


def compute_dnn_accuracy(y_true, y_pred, perc, tranquil=np.nan):
    """Compute the categorical accuracy for the predictions above the
    percentile threshold."""
    max_logits = np.max(y_pred, axis=-1)
    i = np.where(max_logits >= np.percentile(max_logits, 100 - perc))[0]
    met = tf.keras.metrics.CategoricalAccuracy()
    met.update_state(y_true[i, :], y_pred[i, :])
    return met.result().numpy()


def compute_dac_accuracy(y_true, y_pred, abstain):
    """Compute the categorical accuracy the predictions excluding abstentions."""
    cat_pred = tf.math.argmax(y_pred, axis=-1)
    mask = tf.math.not_equal(cat_pred, abstain)
    met = tf.keras.metrics.CategoricalAccuracy()
    met.update_state(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask))
    return met.result().numpy()

# ------------------------------------------------------------------------
# CLASSES
#
#   The following metrics classes are tensorflow compliant.
#
#   See page 390 of Geron, 2019, for a prototype of a metric class. See also,
#   https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric.
# ------------------------------------------------------------------------


class AlphaValue(tf.keras.metrics.Metric):
    """Test

    """
    def __init__(self, loss, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.alpha.assign(self.loss.alpha)

    def result(self):
        return self.alpha

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}


class AbstentionFraction(tf.keras.metrics.Metric):
    """Compute the abstention fraction for an epoch.

    The abstention fraction is the total number of abstentions divided by the
    total number of samples, across the entire epoch. This is not the same
    as the average of batch abstention fractions.

    The computation is done by maintaining running sums of total samples and
    total abstentions made across all batches in an epoch. The running sums
    are reset at the end of each epoch.

    """
    def __init__(self, abstain, **kwargs):
        super().__init__(**kwargs)
        self.abstain = abstain
        self.abstentions = self.add_weight("abstentions", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        cat_pred = tf.math.argmax(y_pred, axis=-1)
        mask = tf.math.equal(cat_pred, self.abstain)

        batch_abstentions = tf.math.count_nonzero(mask)
        batch_total = tf.size(mask)

        self.abstentions.assign_add(tf.cast(batch_abstentions, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.abstentions / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}


class PredictionAccuracy(tf.keras.metrics.Metric):
    """Compute the prediction accuracy for an epoch.

    The prediction accuracy does not include abstentions. The prediction
    accuracy is the total number of correct predictions divided by the
    total number of predictions, across the entire epoch. This is not the
    same as the average of batch prediction accuracies.

    The computation is done by maintaining running sums of total predictions
    and correct predictions made across all batches in an epoch. The running
    sums are reset at the end of each epoch.

    """
    def __init__(self, abstain, **kwargs):
        super().__init__(**kwargs)
        self.abstain = abstain
        self.correct = self.add_weight("correct", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        cat_pred = tf.math.argmax(y_pred, axis=-1)
        cat_true = tf.math.argmax(y_true, axis=-1)

        mask = tf.math.not_equal(cat_pred, self.abstain)
        cat_pred = tf.boolean_mask(cat_pred, mask)
        cat_true = tf.boolean_mask(cat_true, mask)

        batch_correct = tf.math.count_nonzero(tf.math.equal(cat_pred, cat_true))
        batch_total = tf.math.count_nonzero(mask)

        self.correct.assign_add(tf.cast(batch_correct, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.correct / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}


class PredictionLoss(tf.keras.metrics.Metric):
    """Compute the prediction loss for epoch.

    The prediction loss does not include abstentions. Thus, the loss is the
    sample-by-sample cross entropy.

    The prediction loss is the sum predictions losses divided by the total
    number of predictions, across the entire epoch. This is not the same as
    the average of batch prediction losses.

    The computation is done by maintaining running sums of prediction losses
    prediction counts, across the entire epoch. The running sums are reset at
    the end of each epoch.

    """
    def __init__(self, abstain, **kwargs):
        super().__init__(**kwargs)
        self.abstain = abstain
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        predicted = tf.math.argmax(y_pred, axis=-1)

        q = 1 - y_pred[:, -1]
        logq = tf.math.log(q)

        r = tf.boolean_mask(y_pred, y_true)
        logr = tf.math.log(r)

        mask = tf.math.not_equal(predicted, self.abstain)
        loss = tf.boolean_mask(logq - logr, mask)

        batch_count = tf.math.count_nonzero(mask)
        batch_total = tf.math.reduce_sum(loss)

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.total / float(self.count)

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}
