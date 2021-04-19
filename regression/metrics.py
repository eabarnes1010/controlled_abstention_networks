"""Regression with abstention metric classes and functions."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "March 10, 2021"

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprectionWarning)

# ------------------------------------------------------------------------
# CLASSES
#
#   The following metrics classes are tensorflow compliant.
#
#   See page 390 of Geron, 2019, for a prototype of a metric class. See also,
#   https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric.
# ------------------------------------------------------------------------


class AlphaValue(tf.keras.metrics.Metric):
    """Return the value of alpha during training.

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
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.constant(tau, dtype=tf.float64)
        self.abstentions = self.add_weight("abstentions", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mask = tf.math.greater_equal(tf.cast(y_pred[:,1],tf.float64),self.tau)
        
        batch_abstentions = tf.math.count_nonzero(mask)
        batch_total = tf.size(mask)

        self.abstentions.assign_add(tf.cast(batch_abstentions, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.abstentions / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}
    
class Sigma(tf.keras.metrics.Metric):
    """Compute the average prediction sigma across predictions.

    The computation is done by maintaining running sums of total predictions
    and predicted sigmas made across all batches in an epoch. The running
    sums are reset at the end of each epoch.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = self.add_weight("sigma", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        batch_sigma = tf.reduce_sum(y_pred[:,1])
        batch_total = tf.math.count_nonzero(y_pred[:,1])

        self.sigma.assign_add(tf.cast(batch_sigma, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.sigma / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config} 
    
class SigmaCovered(tf.keras.metrics.Metric):
    """Compute the average prediction sigma on covered (non-abstained) predictions.

    The prediction sigma does not include abstentions. The prediction
    sigma is the sum of sigma values divided by the
    total number of predictions, across the entire epoch. This is not the
    same as the average of batch prediction sigmas.

    The computation is done by maintaining running sums of total predictions
    and predicted sigmas made across all batches in an epoch. The running
    sums are reset at the end of each epoch.

    """
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.sigma = self.add_weight("sigma", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mask = tf.math.less(y_pred[:,1],self.tau)        
        sigma_mask = tf.boolean_mask(y_pred[:,1],mask)
        
        batch_sigma = tf.reduce_sum(sigma_mask)
        batch_total = tf.math.count_nonzero(mask)

        self.sigma.assign_add(tf.cast(batch_sigma, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.sigma / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}     
    
class MAE(tf.keras.metrics.Metric):
    """Compute the prediction mean absolute error.

    The computation is done by maintaining running sums of total predictions
    and correct predictions made across all batches in an epoch. The running
    sums are reset at the end of each epoch.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error = self.add_weight("error", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        error = tf.math.abs(tf.math.subtract(y_true[:,0],y_pred[:,0]))        
        batch_error = tf.reduce_sum(error)
        batch_total = tf.math.count_nonzero(error)

        self.error.assign_add(tf.cast(batch_error, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.error / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}    
    
class MAECovered(tf.keras.metrics.Metric):
    """Compute the prediction mean absolute error on covered (non-abstained) predictions.

    The prediction accuracy does not include abstentions. The prediction
    accuracy is the total number of correct predictions divided by the
    total number of predictions, across the entire epoch. This is not the
    same as the average of batch prediction accuracies.

    The computation is done by maintaining running sums of total predictions
    and correct predictions made across all batches in an epoch. The running
    sums are reset at the end of each epoch.

    """
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.error = self.add_weight("error", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mask = tf.math.less(y_pred[:,1],self.tau)        
        error = tf.math.abs(tf.math.subtract(y_true[:,0],y_pred[:,0]))
        error_mask = tf.boolean_mask(error,mask)
        
        batch_error = tf.reduce_sum(error_mask)
        batch_total = tf.math.count_nonzero(mask)

        self.error.assign_add(tf.cast(batch_error, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.error / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}    
    
class Likelihood(tf.keras.metrics.Metric):
    """Compute the prediction mean likelihood on all predictions.

    The computation is done by maintaining running sums of total likelihoods 
    made across all batches in an epoch. The running sums are reset at the 
    end of each epoch.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.likelihood = self.add_weight("likelihood", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mean = tf.cast(y_pred[:,0], tf.float64)
        std = tf.clip_by_value(tf.cast(y_pred[:,1], tf.float64),
                               clip_value_min=1.0e-10,
                               clip_value_max=1.0e10)        
        norm_dist = tfp.distributions.Normal(mean,std)
        
        p = tf.cast(norm_dist.prob(tf.cast(y_true[:,0],tf.float64)), tf.float64)    # likehood of y_true    
        p = tf.clip_by_value(p,clip_value_min=1.0e-10,clip_value_max=1.0e10)
        
        batch_likelihood = tf.reduce_sum(p)
        batch_total = tf.math.count_nonzero(p)

        self.likelihood.assign_add(tf.cast(batch_likelihood, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.likelihood / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}     
    
    
class LikelihoodCovered(tf.keras.metrics.Metric):
    """Compute the prediction mean likelihood on covered (non-abstained) predictions.

    The prediction likelihood does not include abstentions. 

    The computation is done by maintaining running sums of total likelihoods 
    made across all batches in an epoch. The running sums are reset at the 
    end of each epoch.

    """
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.likelihood = self.add_weight("likelihood", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mean = tf.cast(y_pred[:,0], tf.float64)
        std = tf.clip_by_value(tf.cast(y_pred[:,1], tf.float64),
                               clip_value_min=1.0e-10,
                               clip_value_max=1.0e10)        
        norm_dist = tfp.distributions.Normal(mean,std)
        
        p = tf.cast(norm_dist.prob(tf.cast(y_true[:,0],tf.float64)), tf.float64)    # likehood of y_true    
        p = tf.clip_by_value(p,clip_value_min=1.0e-10,clip_value_max=1.0e10)
        
        mask = tf.math.less(y_pred[:,1],self.tau)                                                                
        likelihood_mask = tf.boolean_mask(p,mask)
        
        batch_likelihood = tf.reduce_sum(likelihood_mask)
        batch_total = tf.math.count_nonzero(mask)

        self.likelihood.assign_add(tf.cast(batch_likelihood, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.likelihood / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}     

class LogLikelihood(tf.keras.metrics.Metric):
    """Compute the prediction mean negative log-likelihood on all predictions.

    The computation is done by maintaining running sums of total log-likelihoods 
    made across all batches in an epoch. The running sums are reset at the 
    end of each epoch.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loglikelihood = self.add_weight("loglikelihood", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mean = tf.cast(y_pred[:,0], tf.float64)
        std = tf.clip_by_value(tf.cast(y_pred[:,1], tf.float64),
                               clip_value_min=1.0e-10,
                               clip_value_max=1.0e10)        
        norm_dist = tfp.distributions.Normal(mean,std)
        
        p = tf.cast(norm_dist.prob(tf.cast(y_true[:,0],tf.float64)), tf.float64)    # likehood of y_true    
        p = tf.clip_by_value(p,clip_value_min=1.0e-10,clip_value_max=1.0e10)
        logp = -tf.math.log(p)

        batch_loglikelihood = tf.reduce_sum(logp)
        batch_total = tf.math.count_nonzero(logp)

        self.loglikelihood.assign_add(tf.cast(batch_loglikelihood, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.loglikelihood / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}     
    
class LogLikelihoodCovered(tf.keras.metrics.Metric):
    """Compute the prediction mean negative log-likelihood on covered (non-abstained) predictions.

    The prediction log-likelihood does not include abstentions. 

    The computation is done by maintaining running sums of total lgo-likelihoods 
    made across all batches in an epoch. The running sums are reset at the 
    end of each epoch.

    """
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.loglikelihood = self.add_weight("loglikelihood", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        mean = tf.cast(y_pred[:,0], tf.float64)
        std = tf.clip_by_value(tf.cast(y_pred[:,1], tf.float64),
                               clip_value_min=1.0e-10,
                               clip_value_max=1.0e10)        
        norm_dist = tfp.distributions.Normal(mean,std)
        
        p = tf.cast(norm_dist.prob(tf.cast(y_true[:,0],tf.float64)), tf.float64)    # likehood of y_true    
        p = tf.clip_by_value(p,clip_value_min=1.0e-10,clip_value_max=1.0e10)
        logp = -tf.math.log(p)        
        
        mask = tf.math.less(y_pred[:,1],self.tau)                                                                
        loglikelihood_mask = tf.boolean_mask(logp,mask)
        
        batch_loglikelihood = tf.reduce_sum(loglikelihood_mask)
        batch_total = tf.math.count_nonzero(mask)

        self.loglikelihood.assign_add(tf.cast(batch_loglikelihood, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))
        
    def result(self):
        return self.loglikelihood / self.total

    def get_config(self):
        base_config = super().get_config()
        return{**base_config}      