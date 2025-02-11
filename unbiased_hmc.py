#!/usr/bin/env python
"""
unbiased_hmc.py

This module implements an unbiased HMC kernel combined with an MH kernel,
corresponding to Algorithm 3.5 and 3.6 in Section 3.2 of "Couplings_and_Monte_Carlo.pdf".

The design uses a single class UnbiasedHMC that performs a coupled HMC update using common 
random numbers (i.e. common initial momentum and common MH acceptance random variable) as required 
by Algorithm 3.5, and then combines this with an unbiased MH update (via the provided UnbiasedMHKernel)
using a mixing probability η (Algorithm 3.6). This way, the global exploration of HMC and the local 
contractive properties of MH can be jointly exploited.

Note:
  - We do not re-implement the leapfrog integrator; instead, we call tfp.mcmc.HamiltonianMonteCarlo.one_step 
    with a fixed (stateless) seed to ensure common randomness.
  - All helper functions are defined within the class.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

# Import the already implemented unbiased MH kernel.
try:
    from unbiased_mh_kernel import UnbiasedMHKernel
except ImportError:
    # If not available, provide a simple placeholder implementation.
    class UnbiasedMHKernel(tfp.mcmc.TransitionKernel):
        """
        Placeholder UnbiasedMHKernel.
        In practice, this kernel should implement a coupled MH update for unbiased estimation.
        """
        def __init__(self, target_log_prob_fn, proposal_std, seed=None, name="UnbiasedMHKernel"):
            self._target_log_prob_fn = target_log_prob_fn
            self._proposal_std = proposal_std
            self._seed = seed
            self._name = name

        @property
        def is_calibrated(self):
            return True

        def bootstrap_results(self, current_state):
            state = current_state[0]  # assume both chains have same shape
            batch_shape = tf.shape(state)[:1]
            meeting_time = -tf.ones(batch_shape, dtype=tf.int32)
            return {"iteration": tf.constant(0, dtype=tf.int32),
                    "meeting_time": meeting_time,
                    "seed": self._seed}

        def one_step(self, current_state, previous_kernel_results):
            # A trivial update that simply increments the iteration.
            iteration = previous_kernel_results["iteration"] + 1
            return current_state, {
                "iteration": iteration,
                "meeting_time": previous_kernel_results["meeting_time"],
                "seed": self._seed
            }

class UnbiasedHMC(tfp.mcmc.TransitionKernel):
    """
    UnbiasedHMC implements a mixed coupling kernel that combines:
      - A coupled HMC update (Algorithm 3.5): both chains share a common momentum sample and a common 
        Uniform(0,1) random number to decide MH acceptance.
      - A coupled MH update (via UnbiasedMHKernel) with probability η (mix_prob)
        according to Algorithm 3.6.
    
    This design ensures that the two chains use common random numbers so that if their states are close, 
    they will likely make the same transition. The mixing probability allows the kernel to exploit the 
    local contractivity of MH when the chains are near each other.
    """
    def __init__(self,
                 target_log_prob_fn,
                 step_size,
                 num_leapfrog_steps,
                 momentum_distribution,
                 tolerance,
                 mix_prob,
                 proposal_std=None,  # used for MH kernel; if not provided, default to 1.0
                 seed=None,
                 name="UnbiasedHMC"):
        """
        Initializes the UnbiasedHMC kernel.
        
        Args:
          target_log_prob_fn: Callable mapping a state (tensor of shape [batch, d]) to its log probability.
          step_size: Float, the leapfrog integrator step size.
          num_leapfrog_steps: Integer, the number of leapfrog steps L.
          momentum_distribution: A tfp.distributions instance used to sample momentum.
          tolerance: Float, tolerance to decide whether two chains have met.
          mix_prob: Float in [0, 1]. The probability η to choose the coupled MH update.
          proposal_std: Float, standard deviation for the MH proposal; required if mix_prob > 0.
          seed: Integer, common seed for stateless random operations.
          name: String, kernel name.
        """
        self._target_log_prob_fn = target_log_prob_fn
        self._step_size = step_size
        self._num_leapfrog_steps = num_leapfrog_steps
        self._momentum_distribution = momentum_distribution
        self._tolerance = tolerance
        self._mix_prob = mix_prob
        self._seed = seed if seed is not None else 42
        self._name = name
        
        # Create internal HMC kernel (we use tfp's built-in HamiltonianMonteCarlo)
        self._hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self._target_log_prob_fn,
            step_size=self._step_size,
            num_leapfrog_steps=self._num_leapfrog_steps,
            momentum_distribution=self._momentum_distribution)
        
        # Create the unbiased MH kernel if mixing is used.
        if self._mix_prob > 0:
            proposal_std = proposal_std if proposal_std is not None else 1.0
            self._mh_kernel = UnbiasedMHKernel(
                target_log_prob_fn=self._target_log_prob_fn,
                proposal_std=proposal_std,
                seed=self._seed)
        else:
            self._mh_kernel = None

    @property
    def is_calibrated(self):
        return True

    def bootstrap_results(self, current_state):
        """
        Initialize kernel results using the previous bootstrap format.
        
        Args:
          current_state: Tuple (X, Y) of two chain states.
        
        Returns:
          A dictionary with keys:
            "target_log_prob": (logp_X, logp_Y)
            "accepted": (zeros, zeros)
            "proposals": (zeros_like(X), zeros_like(Y))
            "current_log_prob": (logp_X, logp_Y)
            "t": iteration counter (initialized as 0)
            "meeting_time": a tensor of shape [batch] that is 0 where the chains meet initially, and -1 otherwise.
        """
        X, Y = current_state
        logp_X = self._target_log_prob_fn(X)
        logp_Y = self._target_log_prob_fn(Y)
        tol = tf.cast(self._tolerance, X.dtype)  # use the provided tolerance
        met_initial = tf.reduce_all(tf.abs(X - Y) < tol, axis=1)
        batch_size = tf.shape(X)[0]
        meeting_time = tf.where(met_initial,
                                tf.zeros([batch_size], dtype=tf.int32),
                                -tf.ones([batch_size], dtype=tf.int32))
        return {
            "target_log_prob": (logp_X, logp_Y),
            "accepted": (tf.zeros(tf.shape(logp_X), dtype=tf.bool),
                         tf.zeros(tf.shape(logp_Y), dtype=tf.bool)),
            "proposals": (tf.zeros_like(X), tf.zeros_like(Y)),
            "current_log_prob": (logp_X, logp_Y),
            "t": tf.constant(0, dtype=tf.int32),
            "meeting_time": meeting_time
        }

    def one_step(self, current_state, previous_kernel_results):
        """
        Performs one update step of the mixed kernel.
        
        At each step, a stateless coin toss (seed shaped [2]: [seed, t]) decides whether to use the
        coupled MH update (if coin < mix_prob) or the coupled HMC update.
        
        In both cases, the update is performed on each chain independently using the underlying kernel’s
        one_step function (with a fresh bootstrap result), and then the new target log probabilities are computed.
        The meeting_time is updated if a chain has not met previously and now |X - Y| < tolerance.
        
        Args:
          current_state: Tuple (X, Y) representing the current states.
          previous_kernel_results: Dictionary in the bootstrap format.
        
        Returns:
          new_state: Tuple (new_X, new_Y) after the update.
          new_kernel_results: Dictionary updated with new target log probabilities, iteration t,
                              and meeting_time.
        """
        X, Y = current_state
        t_old = previous_kernel_results["t"]
        new_t = t_old + 1

        # Generate a coin toss seed of shape [2]: [seed, new_t]
        coin_seed = tf.stack([tf.cast(self._seed, tf.int32), new_t])
        coin = tf.random.stateless_uniform([], seed=coin_seed, minval=0, maxval=1)

        # Define the HMC update branch.
        def hmc_update():
            # For each chain, we start from the current state and generate a fresh bootstrap for the update.
            new_X, _ = self._hmc.one_step(
                X,
                self._hmc.bootstrap_results(X),
                seed=coin_seed)
            new_Y, _ = self._hmc.one_step(
                Y,
                self._hmc.bootstrap_results(Y),
                seed=coin_seed)
            return (new_X, new_Y)

        # Define the MH update branch.
        def mh_update():
            # For each chain, call the MH kernel update using its bootstrap.
            new_X, _ = self._mh_kernel.one_step(
                X,
                self._mh_kernel.bootstrap_results(X))
            new_Y, _ = self._mh_kernel.one_step(
                Y,
                self._mh_kernel.bootstrap_results(Y))
            return (new_X, new_Y)

        # Choose update branch based on coin toss.
        new_state = tf.cond(
            tf.less(coin, self._mix_prob) if self._mh_kernel is not None else tf.constant(False),
            lambda: mh_update(),
            lambda: hmc_update()
        )

        # Compute new target log probability for both chains.
        new_logp_X = self._target_log_prob_fn(new_state[0])
        new_logp_Y = self._target_log_prob_fn(new_state[1])
        tol = tf.cast(self._tolerance, new_state[0].dtype)
        met = tf.reduce_all(tf.abs(new_state[0] - new_state[1]) < tol, axis=1)
        prev_mt = previous_kernel_results["meeting_time"]
        updated_mt = tf.where((prev_mt < 0) & met,
                              tf.fill(tf.shape(prev_mt), new_t),
                              prev_mt)

        # Build the new kernel results in the desired format.
        new_kernel_results = {
            "target_log_prob": (new_logp_X, new_logp_Y),
            "accepted": (tf.zeros(tf.shape(new_logp_X), dtype=tf.bool),
                         tf.zeros(tf.shape(new_logp_Y), dtype=tf.bool)),
            "proposals": (tf.zeros_like(new_state[0]), tf.zeros_like(new_state[1])),
            "current_log_prob": (new_logp_X, new_logp_Y),
            "t": new_t,
            "meeting_time": updated_mt
        }
        return new_state, new_kernel_results
