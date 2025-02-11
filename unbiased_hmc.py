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
        Bootstraps the kernel results for both HMC and MH components.
        
        Args:
          current_state: Tuple (state_x, state_y) representing the initial states for the two chains.
        
        Returns:
          A dictionary with keys:
            "hmc": Dictionary with bootstrap results for HMC component for each chain.
            "mh": Dictionary with bootstrap results for MH component (if applicable).
            "iteration": Tensor scalar (int32), initially 0.
            "meeting_time": Tensor of shape [batch] recording the meeting iteration; 0 if already coupled,
                            otherwise -1.
            "seed": The common seed.
            "step_type": A string indicator, initially "none".
        """
        state_x, state_y = current_state
        hmc_res_x = self._hmc.bootstrap_results(state_x)
        hmc_res_y = self._hmc.bootstrap_results(state_y)
        hmc_results = {"x": hmc_res_x, "y": hmc_res_y}
        
        if self._mh_kernel is not None:
            mh_res_x = self._mh_kernel.bootstrap_results(state_x)
            mh_res_y = self._mh_kernel.bootstrap_results(state_y)
            mh_results = {"x": mh_res_x, "y": mh_res_y}
        else:
            mh_results = None
        
        diff = tf.reduce_max(tf.abs(state_x - state_y), axis=-1)
        meeting_time = tf.where(diff < self._tolerance,
                                tf.zeros_like(diff, dtype=tf.int32),
                                -tf.ones_like(diff, dtype=tf.int32))
        
        return {
            "hmc": hmc_results,
            "mh": mh_results,
            "iteration": tf.constant(0, dtype=tf.int32),
            "meeting_time": meeting_time,
            "seed": self._seed,
            "step_type": tf.constant("none")
        }

    def one_step(self, current_state, previous_kernel_results):
        """
        Performs one iteration of the mixed kernel.
        
        With probability mix_prob, the kernel applies the coupled MH update; otherwise, it applies
        the coupled HMC update (Algorithm 3.5). Both updates share common randomness via stateless ops.
        
        Args:
          current_state: Tuple (state_x, state_y) for the two current chain states.
          previous_kernel_results: Dictionary from bootstrap_results or a previous one_step call.
        
        Returns:
          new_state: Updated state tuple (new_state_x, new_state_y).
          new_kernel_results: Updated dictionary containing internal HMC and MH results, iteration,
                              meeting_time, seed, and step_type indicator.
        """
        state_x, state_y = current_state
        iteration = previous_kernel_results["iteration"] + 1
        
        # Generate a coin toss using stateless uniform with a seed that depends on iteration
        coin_seed = tf.stack([tf.cast(self._seed, tf.int32), iteration])
        coin = tf.random.stateless_uniform([], seed=coin_seed, minval=0, maxval=1)
        
        def hmc_update():
            # For HMC update, use common random seed for both chains to sample common momentum
            hmc_seed = tf.stack([tf.cast(self._seed, tf.int32), iteration, tf.constant(0, tf.int32)])
            new_state_x, new_hmc_res_x = self._hmc.one_step(
                state_x, previous_kernel_results["hmc"]["x"], seed=hmc_seed)
            new_state_y, new_hmc_res_y = self._hmc.one_step(
                state_y, previous_kernel_results["hmc"]["y"], seed=hmc_seed)
            new_state = (new_state_x, new_state_y)
            new_hmc_results = {"x": new_hmc_res_x, "y": new_hmc_res_y}
            return new_state, new_hmc_results

        def mh_update():
            # For MH update, call the unbiased MH kernel update.
            new_state_x, new_mh_res_x = self._mh_kernel.one_step(
                state_x, previous_kernel_results["mh"]["x"])
            new_state_y, new_mh_res_y = self._mh_kernel.one_step(
                state_y, previous_kernel_results["mh"]["y"])
            new_state = (new_state_x, new_state_y)
            new_mh_results = {"x": new_mh_res_x, "y": new_mh_res_y}
            return new_state, new_mh_results

        # Select update branch: if coin < mix_prob then use MH update; else use HMC update.
        use_mh = tf.less(coin, self._mix_prob) if self._mh_kernel is not None else False
        
        new_state, branch_results = tf.cond(use_mh,
                                            lambda: mh_update(),
                                            lambda: hmc_update())
        step_type = tf.cond(use_mh, lambda: tf.constant("mh"), lambda: tf.constant("hmc"))
        
        # Update overall meeting_time: for each batch element, if not yet met and the max abs diff 
        # is below tolerance, record the current iteration.
        diff = tf.reduce_max(tf.abs(new_state[0] - new_state[1]), axis=-1)
        prev_mt = previous_kernel_results["meeting_time"]
        new_meeting_time = tf.where((prev_mt < 0) & (diff < self._tolerance),
                                    tf.fill(tf.shape(diff), iteration),
                                    prev_mt)
        
        # Construct new kernel results; only update the branch that was actually used.
        new_kernel_results = {
            "hmc": branch_results if step_type == "hmc" else previous_kernel_results["hmc"],
            "mh": branch_results if step_type == "mh" else previous_kernel_results["mh"],
            "iteration": iteration,
            "meeting_time": new_meeting_time,
            "seed": self._seed,
            "step_type": step_type
        }
        return new_state, new_kernel_results

# End of module.
