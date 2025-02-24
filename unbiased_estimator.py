# File: unbiased_mcmc_estimator.py
"""
File: unbiased_mcmc_estimator.py

This file implements an unbiased MCMC estimator for coupled MH chains following
Algorithm 2.4 (with lag starting at k) described in the attached document "Couplings_and_Monte_Carlo.pdf".

The estimator is defined as:

    Ĥₖ:m = (1/(m - k + 1)) ∑ₗ₌ₖᵐ h(Xₗ) +
           ∑ₜ₌ₖ₊₁^(τ-1) min{1, (t - k)/(m - k + 1)} (h(Xₜ) - h(Yₜ₋₁))

where (Xₜ, Yₜ) is the coupled chain trajectory and τ is the meeting time.
In practice, we first run the coupled chain until every batch has met (or up to a maximum number of iterations),
then select k as (for example) the 90th percentile of the meeting times and set m = m_factor * k.
The full chain samples for chain X and chain Y are stacked (as tf.Tensor) for later TFP-based analysis.

All comments are in English.
"""

import tensorflow as tf
import numpy as np

class UnbiasedMCMCEstimator(object):
    def __init__(self, coupled_kernel, h_fn, quantile=0.9, m_factor=5, max_prelim_iterations=1000):
        """
        Initialize the unbiased MCMC estimator.
        
        Args:
            coupled_kernel: A coupled MCMC kernel instance (e.g., CoupledMetropolisHastingsKernel).
            h_fn: Function mapping a state (tensor of shape [batch, d]) to a scalar evaluation (tensor of shape [batch]).
            quantile: Quantile (e.g., 0.9) to set k from meeting times.
            m_factor: Multiplicative factor to set m = m_factor * k.
            max_prelim_iterations: Maximum iterations allowed in the preliminary run.
        """
        if not (0 < quantile < 1):
            raise ValueError("quantile must be between 0 and 1.")
        self.coupled_kernel = coupled_kernel
        self.h_fn = h_fn
        self.quantile = quantile
        self.m_factor = m_factor
        self.max_prelim_iterations = max_prelim_iterations

    def run(self, initial_state):
        """
        Run the coupled chain and compute the unbiased estimator.
        
        Procedure:
          1. Preliminary run: simulate until each batch has met (meeting_time != -1) or until max_prelim_iterations.
          2. Determine k as the specified quantile (e.g., 90th percentile) of the meeting times; set m = m_factor * k.
          3. Continue simulation (if needed) until m iterations are available.
          4. Compute the estimator:
                 Ĥₖ:m = (1/(m - k + 1)) ∑ₗ₌ₖᵐ h(Xₗ) +
                        ∑ₜ₌ₖ₊₁^(m) min{1, (t - k)/(m - k + 1)} (h(Xₜ) - h(Yₜ₋₁))
          
        Additionally, the full chain samples (chain_X and chain_Y) are stacked as tensors,
        which allows further TFP analyses (e.g., trace plots).
        
        Args:
            initial_state: Tuple (X, Y), each of shape [batch, d].
        
        Returns:
            results: Dictionary with keys:
                "estimator": Tensor of shape [batch] (the unbiased estimates),
                "chain_X": Tensor of shape [total_iterations, batch, d] (trajectory for chain X),
                "chain_Y": Tensor of shape [total_iterations, batch, d] (trajectory for chain Y),
                "meeting_time": Tensor of shape [batch] (meeting times per batch).
        """
        kernel_results = self.coupled_kernel.bootstrap_results(initial_state)
        current_state = initial_state
        state_list = [initial_state]  # store states (each a tuple (X, Y)) at each iteration
        t = 0
        # Preliminary run: run until all batches have met or until maximum iterations reached.
        while t < self.max_prelim_iterations:
            current_state, kernel_results = self.coupled_kernel.one_step(current_state, kernel_results)
            state_list.append(current_state)
            meeting_time = kernel_results["meeting_time"]  # shape [batch]
            if tf.reduce_all(meeting_time >= 0):
                break
            t += 1

        # Convert meeting times to numpy to compute quantile.
        meeting_time_np = meeting_time.numpy()
        k_val = int(np.quantile(meeting_time_np, self.quantile))
        if k_val < 1:
            k_val = 1
        m_val = int(self.m_factor * k_val)

        # Continue simulation until we have at least m_val+1 states.
        while len(state_list) < m_val + 1:
            current_state, kernel_results = self.coupled_kernel.one_step(current_state, kernel_results)
            state_list.append(current_state)

        total_iterations = len(state_list)  # total iterations T >= m_val+1

        # Stack chain trajectories: chain_X and chain_Y will be tensors of shape [T, batch, d]
        chain_X = tf.stack([state[0] for state in state_list], axis=0)
        chain_Y = tf.stack([state[1] for state in state_list], axis=0)

        # Compute evaluations h(X_t) and h(Y_t) along trajectories.
        h_X = [self.h_fn(state[0]) for state in state_list]  # list of T tensors, each shape [batch]
        h_Y = [self.h_fn(state[1]) for state in state_list]  # list of T tensors, each shape [batch]

        # First term: average of h(X_t) from t = k_val to m_val.
        num_avg = m_val - k_val + 1
        stacked_hX = tf.stack(h_X[k_val:], axis=0)  # shape [num_avg, batch]
        term1 = tf.reduce_mean(stacked_hX, axis=0)  # shape [batch]
        
        # Correction term: sum over t from k_val+1 to m_val of weighted differences (h(X_t) - h(Y_{t-1}))
        correction = tf.zeros_like(term1)
        for t_iter in range(k_val + 1, m_val + 1):
            weight = min(1.0, (t_iter - k_val) / num_avg)
            mask = tf.cast(tf.reshape(t_iter < meeting_time, [-1]), h_X[0].dtype)
            diff = h_X[t_iter] - h_Y[t_iter - 1]
            correction += weight * mask * diff

        estimator = term1 + correction

        results = {
            "estimator": estimator,         # shape: [batch]
            "chain_X": chain_X,             # shape: [total_iterations, batch, d]
            "chain_Y": chain_Y,             # shape: [total_iterations, batch, d]
            "meeting_time": meeting_time    # shape: [batch]
        }
        return results
