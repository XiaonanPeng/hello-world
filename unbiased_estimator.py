#!/usr/bin/env python
"""
File: unbiased_mcmc_estimator.py

This file implements an unbiased MCMC estimator for coupled Metropolis-Hastings chains.
The estimator is defined as:

    Ĥₖ:m = (1/(m - k + 1)) * Σₗ₌ₖᵐ h(Xₗ) +
           Σₜ₌ₖ₊₁^(τ-1) min{1, (t - k)/(m - k + 1)} * (h(Xₜ) - h(Yₜ₋₁))

where (Xₜ, Yₜ) is the coupled chain trajectory and τ is the meeting time.
In this implementation, we run a fixed number of MCMC steps using tfp.mcmc.sample_chain,
and then for each batch we compute k and m individually in TensorFlow. Batches that do not meet
or whose m exceeds the available sample iterations are excluded from the final average.

All the computations are performed in TensorFlow to ensure consistency and improved parallel speed.
All comments in this file are in English.
"""

import tensorflow as tf
import tensorflow_probability as tfp

class UnbiasedMCMCEstimator(object):
    def __init__(self, coupled_kernel, h_fn, quantile=0.9, m_factor=5, num_results=1000):
        """
        Initialize the unbiased MCMC estimator.
        
        Args:
            coupled_kernel: A coupled MCMC kernel instance (must inherit from tfp.mcmc.TransitionKernel).
                            The kernel should provide a "meeting_time" in its kernel results.
            h_fn: A function that maps a state (tensor of shape [batch, d]) to a scalar evaluation
                  (tensor of shape [batch]).
            quantile: Quantile used to determine k for each batch (e.g., 0.9).
            m_factor: Multiplicative factor to set m = m_factor * k.
            num_results: The fixed number of MCMC sampling steps (used in tfp.mcmc.sample_chain).
        """
        if not (0 < quantile < 1):
            raise ValueError("quantile must be between 0 and 1.")
        self.coupled_kernel = coupled_kernel
        self.h_fn = h_fn
        self.quantile = quantile
        self.m_factor = m_factor
        self.num_results = num_results

    def run(self, initial_state):
        """
        Run the coupled chain and compute the unbiased estimator.
        
        Procedure:
          1. Use tfp.mcmc.sample_chain to generate a fixed number of samples (num_results). The trace_fn
             records "meeting_time" from the kernel results.
          2. For each batch, determine its meeting time tau as the first iteration t where meeting_time >= 0.
          3. For each valid batch (tau >= 1), compute k as max(1, floor(quantile * tau)); ensure k < tau.
             Then, set m = m_factor * k. If m exceeds num_results, the batch is considered invalid.
          4. Use the chain samples (chain_X and chain_Y) to compute:
                 term1  = (1/(m - k + 1)) * Σₗ₌ₖ^(m) h(Xₗ)
                 correction = Σₜ₌ₖ₊₁^(m) min{1, (t - k)/(m - k + 1)} * (h(Xₜ) - h(Yₜ₋₁))
             For t ≥ tau the difference is set to 0.
          5. The per-batch estimator is computed as term1 + correction.
          6. The final estimator is the mean over all valid batches.
        
        Args:
            initial_state: A tuple (X, Y) where each tensor has shape [batch, d].
        
        Returns:
            results: A dictionary with the following keys:
                "estimator": Scalar; the average estimator over valid batches.
                "per_batch_estimator": Tensor of shape [batch] with each batch's estimator (NaN for invalid batches).
                "chain_X": Tensor of shape [num_results, batch, d] from the coupled chain.
                "chain_Y": Tensor of shape [num_results, batch, d] from the coupled chain.
                "meeting_time": Tensor of meeting times per batch (int32).
                "k_values": Tensor of k values for each batch (int32).
                "m_values": Tensor of m values for each batch (int32).
                "valid_mask": Boolean tensor indicating which batches are valid.
        """
        # Sample a fixed number of steps using tfp.mcmc.sample_chain.
        # initial_state is a tuple (X, Y), so the samples are returned as a tuple.
        # The trace_fn records the "meeting_time" from each kernel result.
        [chain_X, chain_Y], meeting_time_trace = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            current_state=initial_state,
            kernel=self.coupled_kernel,
            trace_fn=lambda _, kr: kr["meeting_time"]
        )
        # chain_X, chain_Y: shape [num_results, batch, d]
        # meeting_time_trace: shape [num_results, batch]
        
        # Determine the meeting time (tau) for each batch using TensorFlow operations.
        # For each batch, tau is the first index t where meeting_time_trace[t] >= 0.
        cond = tf.greater_equal(meeting_time_trace, 0)  # bool tensor, shape [num_results, batch]
        valid_meeting = tf.reduce_any(cond, axis=0)      # shape [batch], bool: True if any meeting occurred.
        # Use tf.argmax on the casted condition; note that if no True exists, argmax returns 0.
        tau_candidate = tf.argmax(tf.cast(cond, tf.int32), axis=0, output_type=tf.int32)
        tau = tf.where(valid_meeting, tau_candidate, -tf.ones_like(tau_candidate))
        
        # Compute per-batch k and m values.
        tau_float = tf.cast(tau, tf.float32)
        k_values = tf.cast(tf.maximum(1.0, tf.floor(self.quantile * tau_float)), tf.int32)
        # Ensure k < tau: if k >= tau, set k = tau - 1.
        k_values = tf.where(tf.greater_equal(k_values, tau), tau - 1, k_values)
        m_values = tf.multiply(self.m_factor, k_values)
        
        # Define overall valid_mask: valid if meeting occurred (tau>=1) and m < num_results.
        valid_mask = tf.logical_and(
            tf.logical_and(tf.greater_equal(tau, 1), tf.less(m_values, self.num_results)),
            valid_meeting
        )
        
        # Compute h(X) and h(Y) evaluations for each time step.
        # For each time t, h_fn operates on chain_X[t] and chain_Y[t] (shape [batch, d]) to yield [batch].
        def compute_h(state):
            # state: Tensor of shape [batch, d]
            return self.h_fn(state)
        
        # Map h_fn over time steps.
        h_X = tf.stack([compute_h(chain_X[t]) for t in range(self.num_results)], axis=0)  # shape [num_results, batch]
        h_Y = tf.stack([compute_h(chain_Y[t]) for t in range(self.num_results)], axis=0)  # shape [num_results, batch]
        
        batch_size = tf.shape(h_X)[1]

        # Compute the unbiased estimator for each batch in a vectorized (or map_fn) manner.
        def compute_estimator_for_batch(i):
            # i is a scalar int32 tensor indicating the batch index.
            tau_i = tau[i]
            k_i = k_values[i]
            m_i = m_values[i]
            valid_i = tf.logical_and(tf.greater_equal(tau_i, 1),
                                     tf.less(m_i, self.num_results))
            # If not valid, return NaN.
            def compute_valid():
                # First term: average of h(X) from t=k_i to t=m_i (inclusive)
                slice_hX = h_X[k_i: m_i + 1, i]
                term1 = tf.reduce_mean(slice_hX)
                
                # Correction term: sum over t from k_i+1 to m_i (only when t < tau_i)
                t_range = tf.range(k_i + 1, m_i + 1)
                # Compute weight for each t: min{1, (t - k_i) / (m_i - k_i + 1)}
                weight = tf.minimum(1.0,
                                    tf.cast(t_range - k_i, tf.float32) /
                                    tf.cast(m_i - k_i + 1, tf.float32))
                # Set mask: only include t < tau_i (if false, value 0)
                mask = tf.cast(tf.less(t_range, tau_i), tf.float32)
                # Compute differences h(X)_t - h(Y)_(t-1) for t in t_range.
                diff = tf.gather(h_X[:, i], t_range) - tf.gather(h_Y[:, i], t_range - 1)
                correction = tf.reduce_sum(weight * mask * diff)
                return term1 + correction
            return tf.cond(valid_i, compute_valid,
                           lambda: tf.constant(float("nan"), dtype=tf.float32))
        
        # Use tf.map_fn over the batch indices (from 0 to batch_size-1)
        per_batch_estimator = tf.map_fn(compute_estimator_for_batch,
                                        tf.range(batch_size),
                                        dtype=tf.float32)
        
        # Compute final estimator as the mean over valid batches.
        valid_estimates = tf.boolean_mask(per_batch_estimator, valid_mask)
        final_estimator = tf.reduce_mean(valid_estimates)
        
        results = {
            "estimator": final_estimator,               # Scalar: average estimator for valid batches.
            "per_batch_estimator": per_batch_estimator,   # Tensor of shape [batch], NaN for invalid batches.
            "chain_X": chain_X,                         # Tensor of shape [num_results, batch, d].
            "chain_Y": chain_Y,                         # Tensor of shape [num_results, batch, d].
            "meeting_time": tau,                        # Tensor of meeting times per batch (int32).
            "k_values": k_values,                       # Tensor of k values per batch (int32).
            "m_values": m_values,                       # Tensor of m values per batch (int32).
            "valid_mask": valid_mask                    # Boolean tensor indicating valid batches.
        }
        return results

if __name__ == '__main__':
    # Example usage:
    # Define your coupled_kernel, h_fn, and initial_state before using the estimator.
    # For example:
    #   coupled_kernel = <YOUR_COUPLED_KERNEL>
    #   h_fn = lambda x: tf.reduce_mean(x, axis=-1)
    #   initial_state = (tf.zeros([8, 2]), tf.zeros([8, 2]))  # Example: batch=8, dimension=2
    #
    # Then:
    #   estimator_obj = UnbiasedMCMCEstimator(coupled_kernel, h_fn, quantile=0.9, m_factor=5, num_results=1000)
    #   results = estimator_obj.run(initial_state)
    #
    # Print desired outputs:
    #   print("Final estimator:", results["estimator"])
    #
    print("Please define coupled_kernel, h_fn, and initial_state to run the estimator.")
