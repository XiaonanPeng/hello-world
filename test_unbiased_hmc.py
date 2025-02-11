#!/usr/bin/env python
"""
test_unbiased_hmc.py

This module uses the standard unittest framework to test the UnbiasedHMC kernel.
We test the kernel on the following three target distributions:
  1. Banana distribution (2-dimensional)
  2. High-dimensional two-mode multimodal normal distribution
  3. Triple-mode Gaussian mixture distribution

The purpose of these tests is to verify that coupled chains meet,
i.e., the recorded meeting_time for every batch element is non-negative.
All comments in the code are provided in English.
"""

import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Import the UnbiasedHMC class defined in unbiased_hmc.py.
from unbiased_hmc import UnbiasedHMC

tfd = tfp.distributions


# Define target distribution log probability functions.
def banana_log_prob(x, b=0.1):
    """
    Banana distribution log probability (2D).
    The first coordinate appears quadratically.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    y2 = x2 + b * (x1**2 - 1.0)
    return -0.5 * (x1**2 + y2**2)


def multimodal_log_prob(x, mu=2.0):
    """
    High-dimensional two-mode multimodal normal distribution log probability.
    Each coordinate's mode is at +mu or -mu, ensuring symmetry.
    """
    center1 = tf.ones_like(x) * mu
    center2 = -tf.ones_like(x) * mu
    log_comp1 = -0.5 * tf.reduce_sum((x - center1)**2, axis=-1)
    log_comp2 = -0.5 * tf.reduce_sum((x - center2)**2, axis=-1)
    max_log = tf.maximum(log_comp1, log_comp2)
    return max_log + tf.math.log(0.5 * tf.exp(log_comp1 - max_log) +
                                 0.5 * tf.exp(log_comp2 - max_log))


def triple_mode_log_prob(x, mu=3.0):
    """
    Triple-mode Gaussian mixture log probability.
    The three modes are located at -mu, 0, and +mu.
    """
    center1 = -mu * tf.ones_like(x)
    center2 = tf.zeros_like(x)
    center3 = mu * tf.ones_like(x)
    log_comp1 = -0.5 * tf.reduce_sum((x - center1)**2, axis=-1)
    log_comp2 = -0.5 * tf.reduce_sum((x - center2)**2, axis=-1)
    log_comp3 = -0.5 * tf.reduce_sum((x - center3)**2, axis=-1)
    max_log = tf.maximum(tf.maximum(log_comp1, log_comp2), log_comp3)
    return max_log + tf.math.log((tf.exp(log_comp1 - max_log) +
                                  tf.exp(log_comp2 - max_log) +
                                  tf.exp(log_comp3 - max_log)) / 3.0)


class TestUnbiasedHMCKernel(unittest.TestCase):
    def setUp(self):
        # Common parameters for all tests
        self.seed = 42
        self.mix_prob = 0.1          # Probability to choose the MH update
        self.tolerance = 1e-3        # Tolerance for chains to be considered as meeting
        self.step_size = 0.1
        self.num_leapfrog_steps = 3
        self.proposal_std = 1.0
        self.max_iter = 200

    def run_chain(self, target_log_prob_fn, d, batch_size, momentum_distribution, distribution_name):
        """
        Simulate a coupled chain using UnbiasedHMC kernel until all batches meet or max_iter is reached.
        
        Args:
          target_log_prob_fn: Callable, log probability function of the target distribution.
          d: Integer, dimension of the state space.
          batch_size: Integer, number of parallel chains.
          momentum_distribution: A tfp.distributions used to sample momentum.
          distribution_name: String, name of the target distribution for logging.
        
        Returns:
          A tuple (meeting_time, iterations) where meeting_time is a numpy array of meeting times.
        """
        # Initialize two chains with independent random starting states
        init_state1 = np.random.randn(batch_size, d).astype(np.float32)
        init_state2 = np.random.randn(batch_size, d).astype(np.float32)
        current_state = (tf.convert_to_tensor(init_state1), tf.convert_to_tensor(init_state2))
        
        # Set kernel parameters
        kernel = UnbiasedHMC(
            target_log_prob_fn=target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            momentum_distribution=momentum_distribution,
            tolerance=self.tolerance,
            mix_prob=self.mix_prob,
            proposal_std=self.proposal_std,
            seed=self.seed
        )
        kernel_results = kernel.bootstrap_results(current_state)
        iteration = 0
        while iteration < self.max_iter:
            meeting_time = kernel_results["meeting_time"].numpy()
            # If all batch elements have met (meeting_time is non-negative), break
            if np.all(meeting_time >= 0):
                break
            current_state, kernel_results = kernel.one_step(current_state, kernel_results)
            iteration += 1

        return kernel_results["meeting_time"].numpy(), iteration

    def test_banana_distribution(self):
        """Test UnbiasedHMC kernel on a 2D Banana distribution."""
        d = 2
        batch_size = 8
        momentum_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
        meeting_time, iters = self.run_chain(banana_log_prob, d, batch_size, momentum_distribution, "Banana")
        # Assert that all chains have met within the maximum iterations.
        self.assertTrue(np.all(meeting_time >= 0),
                        msg=f"Banana distribution: Not all chains met after {iters} iterations, meeting_time: {meeting_time}")

    def test_multimodal_distribution(self):
        """Test UnbiasedHMC kernel on a high-dimensional two-mode multimodal normal distribution."""
        d = 10
        batch_size = 8
        momentum_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
        meeting_time, iters = self.run_chain(multimodal_log_prob, d, batch_size, momentum_distribution, "Multimodal")
        self.assertTrue(np.all(meeting_time >= 0),
                        msg=f"Multimodal distribution: Not all chains met after {iters} iterations, meeting_time: {meeting_time}")

    def test_triple_mode_distribution(self):
        """Test UnbiasedHMC kernel on a triple-mode Gaussian mixture distribution."""
        d = 5
        batch_size = 8
        momentum_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
        meeting_time, iters = self.run_chain(triple_mode_log_prob, d, batch_size, momentum_distribution, "Triple-mode")
        self.assertTrue(np.all(meeting_time >= 0),
                        msg=f"Triple-mode distribution: Not all chains met after {iters} iterations, meeting_time: {meeting_time}")


if __name__ == '__main__':
    unittest.main()
