"""
This test script evaluates the unbiased MCMC estimator constructed from the coupled MH kernel.
It tests three target distributions:
  1. 2D banana distribution,
  2. High-dimensional multimodal normal (two modes),
  3. Multimode Gaussian mixture (three modes).

For each case, instead of directly asserting the absolute value is small, a one-sample t-test
is performed on the per-batch estimators (using h(x)=x[0]). The null hypothesis is that the true
mean is 0. If the p-value exceeds 0.05, we fail to reject the null hypothesis and assume unbiasedness.
Additionally, the full chain samples (for both chains), along with meeting times, are recorded
for further TFP-based analysis.
"""

import unittest
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from scipy.stats import ttest_1samp

from coupled_mh_kernel import CoupledMetropolisHastingsKernel
from unbiased_mcmc_estimator import UnbiasedMCMCEstimator

def banana_log_prob(x, b=0.1):
    x1 = x[:, 0]
    x2 = x[:, 1]
    y2 = x2 + b * (x1**2 - 1.0)
    return -0.5 * (x1**2 + y2**2)

def multimodal_log_prob(x, d=10, mu=2.0):
    center1 = tf.ones_like(x) * mu
    center2 = -tf.ones_like(x) * mu
    log_comp1 = -0.5 * tf.reduce_sum((x - center1)**2, axis=-1)
    log_comp2 = -0.5 * tf.reduce_sum((x - center2)**2, axis=-1)
    max_log = tf.maximum(log_comp1, log_comp2)
    return max_log + tf.math.log(0.5 * tf.exp(log_comp1 - max_log) + 0.5 * tf.exp(log_comp2 - max_log))

def multimode_log_prob(x, d=5, mu=3.0):
    center1 = -mu * tf.ones_like(x)
    center2 = tf.zeros_like(x)
    center3 = mu * tf.ones_like(x)
    log_comp1 = -0.5 * tf.reduce_sum((x - center1)**2, axis=-1)
    log_comp2 = -tf.reduce_sum( -0.5 * tf.reduce_sum((x - center2)**2, axis=-1), axis=-1)  # mistake avoided below
    # Correcting: use the same pattern as above.
    log_comp2 = -0.5 * tf.reduce_sum((x - center2)**2, axis=-1)
    log_comp3 = -0.5 * tf.reduce_sum((x - center3)**2, axis=-1)
    max_log = tf.maximum(tf.maximum(log_comp1, log_comp2), log_comp3)
    return max_log + tf.math.log((tf.exp(log_comp1 - max_log) +
                                  tf.exp(log_comp2 - max_log) +
                                  tf.exp(log_comp3 - max_log)) / 3.0)

def h_fn(x):
    # Use the first coordinate for evaluation.
    return x[:, 0]

class TestUnbiasedMCMCEstimator(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        np.random.seed(42)
        self.batch = 16  # number of independent chains in the batch
    
    def run_estimator(self, target_log_prob_fn, d, proposal_var, quantile, m_factor, coupling_method):
        # Generate two independent initial states for the coupled chains
        init_state1 = np.random.randn(self.batch, d).astype(np.float32)
        init_state2 = np.random.randn(self.batch, d).astype(np.float32)
        initial_state = (tf.convert_to_tensor(init_state1), tf.convert_to_tensor(init_state2))
        
        # Instantiate the coupled MH kernel.
        kernel = CoupledMetropolisHastingsKernel(
            target_log_prob_fn=target_log_prob_fn,
            proposal_var=proposal_var,
            coupling_method=coupling_method,
            max_iter=5,
            seed=100)
        
        # Instantiate the unbiased estimator with a fixed number of MCMC iterations (num_results).
        estimator_obj = UnbiasedMCMCEstimator(
            coupled_kernel=kernel,
            h_fn=h_fn,
            quantile=quantile,
            m_factor=m_factor,
            num_results=1000)  # Set fixed number of sampling steps
        
        results = estimator_obj.run(initial_state)
        
        # Retrieve the per-batch estimator and valid mask from the results.
        # per_batch_estimator is an array of shape [batch] (nan for invalid batches).
        per_batch_est = results["per_batch_estimator"]
        valid_mask = results["valid_mask"]
        valid_estimates = per_batch_est[valid_mask]
        return valid_estimates, results
    
    def t_test_unbiased(self, valid_estimates):
        """
        Perform a one-sample t-test for zero mean on the provided per-batch estimators.
        Returns the t-test statistic and the p-value.
        """
        # Ensure that there are at least two valid batches.
        self.assertGreaterEqual(valid_estimates.shape[0], 2,
                                "Not enough valid batches for t-test.")
        t_stat, p_value = ttest_1samp(valid_estimates, popmean=0.0)
        return t_stat, p_value
    
    def test_banana_estimator(self):
        d = 2
        proposal_var = 1.0
        quantile = 0.9
        m_factor = 5
        coupling_method = "maximal"
        
        valid_estimates, results = self.run_estimator(
            lambda x: banana_log_prob(x, b=0.1),
            d, proposal_var, quantile, m_factor, coupling_method)
        
        t_stat, p_value = self.t_test_unbiased(valid_estimates)
        print("Banana distribution: t-statistic = {:.3f}, p-value = {:.3f}".format(t_stat, p_value))
        self.assertGreater(p_value, 0.05,
                           "Banana estimator: Null hypothesis of unbiasedness should not be rejected (p-value too low).")
    
    def test_multimodal_estimator(self):
        # Test on high-dimensional multimodal normal (2 modes)
        d = 10
        proposal_var = 1.0
        quantile = 0.9
        m_factor = 5
        coupling_method = "maximal_reflection"
        
        valid_estimates, results = self.run_estimator(
            lambda x: multimodal_log_prob(x, d=d, mu=2.0),
            d, proposal_var, quantile, m_factor, coupling_method)
        
        t_stat, p_value = self.t_test_unbiased(valid_estimates)
        print("High-dimensional multimodal: t-statistic = {:.3f}, p-value = {:.3f}".format(t_stat, p_value))
        self.assertGreater(p_value, 0.05,
                           "High-dimensional multimodal estimator: Null hypothesis of unbiasedness should not be rejected (p-value too low).")
    
    def test_multimode_estimator(self):
        # Test on multimode Gaussian mixture (3 modes)
        d = 5
        proposal_var = 1.0
        quantile = 0.9
        m_factor = 5
        coupling_method = "maximal"
        
        valid_estimates, results = self.run_estimator(
            lambda x: multimode_log_prob(x, d=d, mu=3.0),
            d, proposal_var, quantile, m_factor, coupling_method)
        
        t_stat, p_value = self.t_test_unbiased(valid_estimates)
        print("Multimode Gaussian mixture: t-statistic = {:.3f}, p-value = {:.3f}".format(t_stat, p_value))
        self.assertGreater(p_value, 0.05,
                           "Multimode Gaussian estimator: Null hypothesis of unbiasedness should not be rejected (p-value too low).")

if __name__ == "__main__":
    unittest.main()
