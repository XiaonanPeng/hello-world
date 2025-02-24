"""
This file implements an unbiased MCMC estimator for coupled Metropolis-Hastings chains.
The estimator is defined as:

    Ĥₖ:m = (1/(m - k + 1)) * Σₗ₌ₖᵐ h(Xₗ) +
           Σₜ₌ₖ₊₁^(τ-1) min{1, (t - k)/(m - k + 1)} * (h(Xₜ) - h(Yₜ₋₁))

where (Xₜ, Yₜ) is the coupled chain trajectory and τ is the meeting time.
In this implementation, we run a fixed number of MCMC steps using tfp.mcmc.sample_chain,
and then for each batch we compute k and m individually. If a batch does not meet or its m
exceeds the available sample iterations, that batch is excluded from the final average.

All comments in this file are in English.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class UnbiasedMCMCEstimator(object):
    def __init__(self, coupled_kernel, h_fn, quantile=0.9, m_factor=5, num_results=1000):
        """
        Initialize the unbiased MCMC estimator.
        
        Args:
            coupled_kernel: A coupled MCMC kernel instance (must inherit from tfp.mcmc.TransitionKernel).
                            The kernel should also provide a "meeting_time" in its kernel results.
            h_fn: A function that maps a state (tensor of shape [batch, d]) to a scalar evaluation
                  (tensor of shape [batch]).
            quantile: Quantile to determine k for each batch (e.g., 0.9).
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
          1. Use tfp.mcmc.sample_chain to generate a fixed number of samples (num_results). A trace_fn
             extracts "meeting_time" from the kernel's trace at each iteration.
          2. For each batch, determine its meeting time tau as the first iteration t where meeting_time >= 0.
          3. For each valid batch (tau >= 1), determine k_i as max(1, int(quantile * tau)); ensure k_i < tau.
             Then set m_i = m_factor * k_i. If m_i exceeds num_results, that batch is marked invalid.
          4. Compute the estimator for each valid batch:
                 term1 = (1/(m_i - k_i + 1)) * sum_{l=k_i}^{m_i} h(X_l)
                 correction = sum_{t=k_i+1}^{m_i} min{1, (t - k_i)/(m_i - k_i + 1)} * (h(X_t) - h(Y_{t-1}))
             For t >= tau the difference (h(X_t) - h(Y_{t-1})) is set to zero.
          5. Average the per-batch estimators over all valid batches to yield the final estimator.
        
        Args:
            initial_state: A tuple (X, Y) where each tensor has shape [batch, d].
        
        Returns:
            results: A dictionary with the following keys:
                "estimator": Scalar; the average estimator over valid batches.
                "per_batch_estimator": Array of shape [batch] with each batch's estimator (nan for invalid batches).
                "chain_X": Tensor of shape [num_results, batch, d] from the coupled chain.
                "chain_Y": Tensor of shape [num_results, batch, d] from the coupled chain.
                "meeting_time": Numpy array of meeting times (tau) per batch.
                "k_values": Numpy array of k values for each batch.
                "m_values": Numpy array of m values for each batch.
                "valid_mask": Boolean numpy array indicating which batches are valid.
        """
        # Sample a fixed number of steps using tfp.mcmc.sample_chain.
        # Note: initial_state is a tuple (X, Y), so the samples are returned as a tuple.
        # The trace_fn records the "meeting_time" from the kernel result.
        [chain_X, chain_Y], meeting_time_trace = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            current_state=initial_state,
            kernel=self.coupled_kernel,
            trace_fn=lambda _, kr: kr["meeting_time"]
        )
        # chain_X and chain_Y have shape [num_results, batch, d]
        # meeting_time_trace has shape [num_results, batch]

        # Convert meeting_time_trace to a numpy array for per-batch processing.
        meeting_time_trace_np = meeting_time_trace.numpy()  # Shape: [num_results, batch]
        num_results, batch_size = meeting_time_trace_np.shape
        
        # Determine the meeting time tau for each batch as the first iteration t with meeting_time >= 0.
        tau = np.full((batch_size,), -1, dtype=int)
        for i in range(batch_size):
            t_valid = np.where(meeting_time_trace_np[:, i] >= 0)[0]
            if t_valid.size > 0:
                tau[i] = int(t_valid[0])
            # If no meeting occurred for this batch, tau[i] remains -1.

        # Compute per-batch k and m values based on tau.
        k_values = np.full((batch_size,), -1, dtype=int)
        m_values = np.full((batch_size,), -1, dtype=int)
        valid_mask = np.zeros((batch_size,), dtype=bool)
        for i in range(batch_size):
            if tau[i] < 1:  # Skip if meeting did not occur or tau is too small.
                continue
            # Set k_i as max(1, int(quantile * tau[i])). Ensure k_i < tau.
            k_i = max(1, int(self.quantile * tau[i]))
            if k_i >= tau[i]:
                k_i = tau[i] - 1
            m_i = int(self.m_factor * k_i)
            # If m_i exceeds available iterations, mark this batch as invalid.
            if m_i >= self.num_results:
                continue
            k_values[i] = k_i
            m_values[i] = m_i
            valid_mask[i] = True

        # Compute h(X) and h(Y) evaluations for each time step.
        # chain_X and chain_Y: [num_results, batch, d]
        # h_fn expects input of shape [batch, d] and outputs shape [batch].
        h_X_list = []
        h_Y_list = []
        for t in range(self.num_results):
            h_X_list.append(self.h_fn(chain_X[t]))  # Output shape: [batch]
            h_Y_list.append(self.h_fn(chain_Y[t]))  # Output shape: [batch]
        # Stack into tensors of shape [num_results, batch].
        h_X = tf.stack(h_X_list, axis=0)
        h_Y = tf.stack(h_Y_list, axis=0)
        h_X_np = h_X.numpy()  # Convert to numpy for easier per-batch processing.
        h_Y_np = h_Y.numpy()

        # Compute the unbiased estimator for each valid batch.
        per_batch_estimator = np.full((batch_size,), np.nan, dtype=np.float32)
        for i in range(batch_size):
            if not valid_mask[i]:
                continue  # Skip batch if not valid.
            k_i = k_values[i]
            m_i = m_values[i]
            tau_i = tau[i]
            # First term: average of h(X) from t=k_i to t=m_i.
            term1 = np.mean(h_X_np[k_i : m_i + 1, i])
            # Correction term: sum from t=k_i+1 to m_i with weight = min{1, (t - k_i)/(m_i - k_i + 1)}
            # Only include t values before tau (i.e., before meeting).
            correction = 0.0
            num_avg = m_i - k_i + 1
            for t_iter in range(k_i + 1, m_i + 1):
                if t_iter >= tau_i:
                    continue
                weight = min(1.0, (t_iter - k_i) / num_avg)
                diff = h_X_np[t_iter, i] - h_Y_np[t_iter - 1, i]
                correction += weight * diff
            per_batch_estimator[i] = term1 + correction

        # Compute the final estimator by averaging the estimates over all valid batches.
        if np.any(valid_mask):
            final_estimator = np.mean(per_batch_estimator[valid_mask])
        else:
            final_estimator = np.nan

        results = {
            "estimator": final_estimator,           # Scalar: average estimator for valid batches.
            "per_batch_estimator": per_batch_estimator,  # Array [batch]; nan for invalid batches.
            "chain_X": chain_X,                     # Tensor of shape [num_results, batch, d].
            "chain_Y": chain_Y,                     # Tensor of shape [num_results, batch, d].
            "meeting_time": tau,                    # Numpy array of meeting times per batch.
            "k_values": k_values,                   # Numpy array of k values for each batch.
            "m_values": m_values,                   # Numpy array of m values for each batch.
            "valid_mask": valid_mask                # Boolean array indicating valid batches.
        }
        return results
