#!/usr/bin/env python
"""
test_unbiased_multinomial_hmc.py

This test module uses the standard unittest framework to test the
UnbiasedMultinomialHMC kernel on three target distributions:
    1. Banana distribution (2D)
    2. Two-mode Multimodal Normal distribution
    3. Triple-mode Gaussian Mixture distribution

The tests verify that the coupled chains meet (i.e. meeting_time becomes non-negative).
"""

import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from unbiased_multinomial_hmc import UnbiasedMultinomialHMC

tfd = tfp.distributions

def banana_log_prob(x, b=0.1):
    """
    Banana distribution log probability (2D).
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    y2 = x2 + b * (tf.square(x1) - 1.0)
    return -0.5 * (tf.square(x1) + tf.square(y2))

def multimodal_log_prob(x, mu=2.0):
    """
    Two-mode Multimodal Normal distribution log probability.
    """
    center1 = tf.ones_like(x) * mu
    center2 = -tf.ones_like(x) * mu
    log_comp1 = -0.5 * tf.reduce_sum(tf.square(x - center1), axis=-1)
    log_comp2 = -0.5 * tf.reduce_sum(tf.square(x - center2), axis=-1)
    max_log = tf.maximum(log_comp1, log_comp2)
    return max_log + tf.math.log(0.5 * tf.exp(log_comp1 - max_log) + 0.5 * tf.exp(log_comp2 - max_log))

def triple_mode_log_prob(x, mu=3.0):
    """
    Triple-mode Gaussian Mixture distribution log probability.
    """
    center1 = -mu * tf.ones_like(x)
    center2 = tf.zeros_like(x)
    center3 = mu * tf.ones_like(x)
    log_comp1 = -0.5 * tf.reduce_sum(tf.square(x - center1), axis=-1)
    log_comp2 = -0.5 * tf.reduce_sum(tf.square(x - center2), axis=-1)
    log_comp3 = -0.5 * tf.reduce_sum(tf.square(x - center3), axis=-1)
    max_log = tf.maximum(tf.maximum(log_comp1, log_comp2), log_comp3)
    return max_log + tf.math.log((tf.exp(log_comp1 - max_log) + tf.exp(log_comp2 - max_log) + tf.exp(log_comp3 - max_log)) / 3.0)

class TestUnbiasedMultinomialHMC(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.mix_prob = 0.1          # Probability to choose MH update.
        self.tolerance = 1e-3        # Meeting tolerance.
        self.step_size = 0.1
        self.num_leapfrog_steps = 5
        self.proposal_std = 1.0
        self.max_iter = 200

    def run_chain(self, target_log_prob_fn, d, batch_size, momentum_distribution, distribution_name):
        print(f"Testing {distribution_name} distribution with d={d}, batch_size={batch_size}")
        
        init_state1 = np.random.randn(batch_size, d).astype(np.float32)
        init_state2 = np.random.randn(batch_size, d).astype(np.float32)
        current_state = (tf.convert_to_tensor(init_state1), tf.convert_to_tensor(init_state2))
        
        kernel = UnbiasedMultinomialHMC(
            target_log_prob_fn=target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            momentum_distribution=momentum_distribution,
            tolerance=self.tolerance,
            mix_prob=self.mix_prob,
            coupling_method="maximal",   # Can be "maximal" or "w2"
            reg=1e-1,
            proposal_std=self.proposal_std,
            seed=self.seed
        )
        kernel_results = kernel.bootstrap_results(current_state)
        iteration = 0
        while iteration < self.max_iter:
            mt = kernel_results["meeting_time"].numpy()
            if np.all(mt >= 0):
                print(f"{distribution_name}: All batches met at iteration {iteration}, meeting_time: {mt}")
                break
            current_state, kernel_results = kernel.one_step(current_state, kernel_results)
            iteration += 1
        else:
            print(f"{distribution_name}: Not all chains met within {self.max_iter} iterations. Final meeting_time: {mt}")
        return mt, iteration

    def test_banana_distribution(self):
        d = 2
        batch_size = 8
        momentum_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
        mt, iters = self.run_chain(banana_log_prob, d, batch_size, momentum_distribution, "Banana")
        self.assertTrue(np.all(mt >= 0), f"Banana: meeting_time not achieved in {iters} iterations, mt: {mt}")

    def test_multimodal_distribution(self):
        d = 10
        batch_size = 8
        momentum_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
        mt, iters = self.run_chain(multimodal_log_prob, d, batch_size, momentum_distribution, "Multimodal")
        self.assertTrue(np.all(mt >= 0), f"Multimodal: meeting_time not achieved in {iters} iterations, mt: {mt}")

    def test_triple_mode_distribution(self):
        d = 5
        batch_size = 8
        momentum_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
        mt, iters = self.run_chain(triple_mode_log_prob, d, batch_size, momentum_distribution, "Triple-mode")
        self.assertTrue(np.all(mt >= 0), f"Triple-mode: meeting_time not achieved in {iters} iterations, mt: {mt}")

if __name__ == '__main__':
    unittest.main()
