"""
This module implements a coupled multinomial Hamiltonian Monte Carlo (HMC) kernel.
The update mechanism is as follows:

  - At each update, the kernel performs a coupled update by randomly choosing
    (via a coin toss) between a coupled multinomial HMC update and a coupled Metropolis-Hastings (MH) update.
  - The multinomial HMC update simulates a full trajectory using Hamiltonian dynamics,
    following the procedure in Section 2.2 of the attached paper:
      * A number L_f is uniformly sampled from {0, 1, ..., L} as the number of forward leapfrog steps.
      * The number of backward steps is then L - L_f.
      * The forward trajectory is obtained by integrating from (q0, p0) for L_f steps;
        the backward trajectory is obtained by integrating from (q0, -p0) for L - L_f steps.
      * The backward trajectory (excluding the duplicate q0) is reversed and concatenated with the forward trajectory.
  - When the current time step t is less than a specified lag, only the X-chain is updated (Y stays unchanged)
    so that both chains use identical marginal updates during the lag period.
  - With probability mix_prob a coupled MH update is performed using the external MH kernel,
    called coupledRWMHKernel (imported from coupled_mh_kernel.py).

All comments are in English.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from mlfinlib.ssmlib.unbiased_mcmc.couplings import (
    maximal_multinomial_coupling,
    w2_multinomial_coupling,
)
from mlfinlib.ssmlib.unbiased_mcmc.unbiased_mh import CoupledRWMHKernel

tfd = tfp.distributions


def efficient_leapfrog_trajectory(q0, p0, num_steps, step_size, target_log_prob_fn):
    """
    Efficient leapfrog integrator that computes gradients only L+1 times for a trajectory of length L+1.

    Args:
      q0: Initial position tensor of shape [d]
      p0: Initial momentum tensor of shape [d]
      num_steps: Number of leapfrog steps (integer scalar)
      step_size: Integration step size.
      target_log_prob_fn: Function that computes log probability.
        (Expects input shape [1, d])

    Returns:
      traj_q: Tensor of positions with shape [num_steps+1, d]
      traj_p: Tensor of momenta with shape [num_steps+1, d]
      traj_logp: Tensor of log probabilities, shape [num_steps+1]
    """
    d = tf.shape(q0)[0]

    # Initialize TensorArrays for trajectory storage.
    TA_q = tf.TensorArray(dtype=q0.dtype, size=num_steps + 1)
    TA_p = tf.TensorArray(dtype=p0.dtype, size=num_steps + 1)
    TA_logp = tf.TensorArray(dtype=q0.dtype, size=num_steps + 1)

    # Evaluate initial log probability and gradient at q0.
    q0_exp = tf.expand_dims(q0, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(q0_exp)
        logp0 = target_log_prob_fn(q0_exp)[0]
    grad_U = -tape.gradient(logp0, q0_exp)[0]

    # Initial half-step update for momentum.
    p_half = p0 - 0.5 * step_size * grad_U

    # Write the initial state (q0, p_half, logp0).
    TA_q = TA_q.write(0, q0)
    TA_p = TA_p.write(0, p_half)
    TA_logp = TA_logp.write(0, logp0)

    # Set current state for the loop: note that we already updated p with a half-step.
    q_current = q0
    p_current = p_half

    i = tf.constant(0, dtype=tf.int32)

    def cond(i, q, p, TA_q, TA_p, TA_logp):
        return tf.less(i, num_steps)

    def body(i, q, p, TA_q, TA_p, TA_logp):
        # Full step for position: q <- q + step_size * p.
        q_new = q + step_size * p
        q_new_exp = tf.expand_dims(q_new, axis=0)
        # Compute new log probability and gradient at q_new.
        with tf.GradientTape() as tape:
            tape.watch(q_new_exp)
            logp_new = target_log_prob_fn(q_new_exp)[0]
        grad_U_new = -tape.gradient(logp_new, q_new_exp)[0]
        # Update momentum:
        # For intermediate steps use a full step; for final step use a half-step update.
        p_new = tf.cond(
            tf.less(i, num_steps - 1), lambda: p - step_size * grad_U_new, lambda: p - 0.5 * step_size * grad_U_new
        )
        # Write the new state into TensorArrays.
        TA_q = TA_q.write(i + 1, q_new)
        TA_p = TA_p.write(i + 1, p_new)
        TA_logp = TA_logp.write(i + 1, logp_new)
        return i + 1, q_new, p_new, TA_q, TA_p, TA_logp

    # Run the while loop for num_steps iterations.
    _, _, _, TA_q_final, TA_p_final, TA_logp_final = tf.while_loop(
        cond, body, loop_vars=(i, q_current, p_current, TA_q, TA_p, TA_logp)
    )

    traj_q = TA_q_final.stack()  # Shape: [num_steps+1, d]
    traj_p = TA_p_final.stack()  # Shape: [num_steps+1, d]
    traj_logp = TA_logp_final.stack()  # Shape: [num_steps+1]

    return traj_q, traj_p, traj_logp


def simulate_direction(q0, p0, num_steps, step_size, target_log_prob_fn):
    """
    Simulate a trajectory for one sample in one integration direction using efficient leapfrog.

    Args:
      q0: Initial position tensor [d]
      p0: Initial momentum tensor [d]
      num_steps: Number of leapfrog steps (integer scalar)
      step_size: Integration step size.
      target_log_prob_fn: Function to compute log probability.

    Returns:
      traj_q: Tensor of positions with shape [num_steps+1, d]
      traj_p: Tensor of momenta with shape [num_steps+1, d]
      traj_logp: Tensor of log probabilities with shape [num_steps+1]
    """

    def no_step():
        # If no steps are taken, simply return the initial state.
        q0_exp = tf.expand_dims(q0, axis=0)
        logp0 = tf.stop_gradient(target_log_prob_fn(q0_exp))
        return q0_exp, tf.expand_dims(p0, axis=0), logp0

    traj_q, traj_p, traj_logp = tf.cond(
        tf.equal(num_steps, 0),
        no_step,
        lambda: efficient_leapfrog_trajectory(q0, p0, num_steps, step_size, target_log_prob_fn),
    )
    return traj_q, traj_p, traj_logp


def simulate_trajectory(q0_batch, p0_batch, L, step_size, target_log_prob_fn):
    """
    Simulate trajectories for a batch of samples using dynamic forward/backward integration.

    For each sample:
      - Randomly choose the number of forward integration steps (Lf) from {0,...,L}.
      - Compute backward integration steps as Lb = L - Lf.
      - Simulate forward trajectory using initial momentum p0.
      - Simulate backward trajectory using negative momentum (-p0).
      - Remove the duplicate initial state from the backward trajectory (if any),
        reverse the backward trajectory, and concatenate it with the forward trajectory.
        This guarantees a total trajectory length of L+1.

    Args:
      q0_batch: Tensor of initial positions, shape [batch, d]
      p0_batch: Tensor of initial momenta, shape [batch, d]
      L: Integer scalar representing the upper bound for forward steps; total steps = L+1.
      step_size: Integration step size.
      target_log_prob_fn: Function to compute log probability.

    Returns:
      traj_q_batch: Tensor of positions, shape [batch, L+1, d]
      traj_p_batch: Tensor of momenta, shape [batch, L+1, d]
      traj_logp_batch: Tensor of log probabilities, shape [batch, L+1]
    """

    def simulate_single(sample):
        # Unpack the individual sample's initial state and momentum (both shape [d]).
        q0, p0 = sample
        d = tf.shape(q0)[0]
        # Randomly choose forward integration steps Lf from {0, 1, ..., L}.
        Lf = tf.random.uniform(shape=[], minval=0, maxval=L + 1, dtype=tf.int32)
        # Backward integration steps Lb = L - Lf.
        Lb = L - Lf

        # Simulate forward trajectory with positive momentum p0.
        forward_q, forward_p, forward_logp = simulate_direction(q0, p0, Lf, step_size, target_log_prob_fn)
        # Simulate backward trajectory with negative momentum -p0.
        backward_q, backward_p, backward_logp = simulate_direction(q0, -p0, Lb, step_size, target_log_prob_fn)

        # Remove duplicate initial state from the backward trajectory (if any) and reverse its order.
        def get_backward():
            return (
                tf.reverse(backward_q[1:], axis=[0]),
                tf.reverse(backward_p[1:], axis=[0]),
                tf.reverse(backward_logp[1:], axis=[0]),
            )

        q_back, p_back, logp_back = tf.cond(
            tf.equal(Lb, 0),
            lambda: (
                tf.zeros([0, d], dtype=q0.dtype),
                tf.zeros([0, d], dtype=p0.dtype),
                tf.zeros([0], dtype=forward_logp.dtype),
            ),
            get_backward,
        )

        # Concatenate backward (reversed) trajectory with forward trajectory.
        traj_q = tf.concat([q_back, forward_q], axis=0)  # Shape: [L+1, d]
        traj_p = tf.concat([p_back, forward_p], axis=0)  # Shape: [L+1, d]
        traj_logp = tf.concat([logp_back, forward_logp], axis=0)  # Shape: [L+1]
        return traj_q, traj_p, traj_logp

    # Process each sample in the batch via tf.map_fn.
    traj_q_batch, traj_p_batch, traj_logp_batch = tf.map_fn(
        simulate_single, (q0_batch, p0_batch), dtype=(q0_batch.dtype, p0_batch.dtype, tf.float32)
    )

    return traj_q_batch, traj_p_batch, traj_logp_batch


class UnbiasedMultinomialHMC(tfp.mcmc.TransitionKernel):
    """
    UnbiasedMultinomialHMC implements a coupled multinomial HMC kernel.

    Update mechanism:
      - At each step, the kernel chooses between a coupled multinomial HMC update and a coupled MH update.
      - When a coupled multinomial HMC update is chosen, a full trajectory is simulated as detailed in simulate_trajectory().
      - When the current time step t is less than lag, only the X-chain is updated (the Y-chain remains unchanged).
      - The coupled MH update is performed by calling the external MH kernel (coupledRWMHKernel) using the tuple (X, Y).
    """

    def __init__(
        self,
        target_log_prob_fn,
        step_size,
        L,  # Maximum leapfrog steps
        momentum_distribution,
        tolerance,
        mix_prob,
        lag,  # Number of initial steps during which Y is not updated
        coupling_method="maximal",  # Options: "maximal" or "w2"
        reg=1e-1,
        proposal_var=None,  # Proposal variance for the MH update
        seed=None,
        name="UnbiasedMultinomialHMC",
    ):
        """
        Initialize the kernel.

        Args:
            target_log_prob_fn: Callable mapping [batch, d] to log probability.
            step_size: Float leapfrog step size.
            L: Integer maximum leapfrog steps.
            momentum_distribution: A tfp.distributions instance for sampling momentum.
            tolerance: Float threshold for meeting.
            mix_prob: Float in [0,1] probability of performing a coupled MH update.
            lag: Integer, number of initial steps in which the Y-chain is not updated.
            coupling_method: String, either "maximal" or "w2" for intra-trajectory coupling.
            reg: Regularization parameter for W2 coupling.
            proposal_var: Float proposal variance for the MH update.
            seed: Optional integer random seed.
            name: Kernel name.
        """
        self._target_log_prob_fn = target_log_prob_fn
        self._step_size = step_size
        self._L = L
        self._momentum_distribution = momentum_distribution
        self._tolerance = tolerance
        self._mix_prob = mix_prob
        self._lag = lag
        self._coupling_method = coupling_method
        self._reg = reg
        self._seed = seed if seed is not None else 42
        self._name = name

        if self._mix_prob > 0:
            proposal_var = proposal_var if proposal_var is not None else 1.0
            # Use the externally defined coupled MH kernel.
            self._mh_kernel = CoupledRWMHKernel(
                self._target_log_prob_fn,
                proposal_var,
                coupling_method="maximal",
                seed=self._seed,
                tolerance=self._tolerance,
                lag=0,
            )
        else:
            self._mh_kernel = None

    @property
    def is_calibrated(self):
        return True

    def bootstrap_results(self, current_state):
        """
        Bootstraps kernel results.

        Args:
            current_state: Tuple (X, Y) of current states, each of shape [batch, d].

        Returns:
            A dictionary containing:
              - "target_log_prob": (logp_X, logp_Y)
              - "accepted": (False, False) tensors.
              - "proposals": (zeros, zeros) of the same shape as X and Y.
              - "current_log_prob": (logp_X, logp_Y)
              - "t": A scalar integer iteration counter.
              - "meeting_time": A tensor of shape [batch] initialized to -1 if chains have not met.
        """
        X, Y = current_state
        logp_X = self._target_log_prob_fn(X)
        logp_Y = self._target_log_prob_fn(Y)
        tol = tf.cast(self._tolerance, X.dtype)
        met_initial = tf.reduce_all(tf.abs(X - Y) < tol, axis=1)
        batch_size = tf.shape(X)[0]
        meeting_time = tf.where(
            met_initial, tf.zeros([batch_size], dtype=tf.int32), -tf.ones([batch_size], dtype=tf.int32)
        )
        Y_history = tf.tile(tf.expand_dims(Y, axis=0), [self._lag, 1, 1])
        return {
            "target_log_prob": (logp_X, logp_Y),
            "accepted": (tf.zeros(tf.shape(logp_X), dtype=tf.bool), tf.zeros(tf.shape(logp_Y), dtype=tf.bool)),
            "proposals": (tf.zeros_like(X), tf.zeros_like(Y)),
            "current_log_prob": (logp_X, logp_Y),
            "t": tf.constant(0, dtype=tf.int32),
            "meeting_time": meeting_time,
            "Y_history": Y_history,
        }

    def one_step(self, current_state, previous_kernel_results):
        """
        Performs one coupled update step.

        At each step, a coin toss (using a stateless seed) chooses between:
          - A coupled HMC update (simulating a full multinomial HMC trajectory with shared momentum
            and using the chosen intra-trajectory coupling strategy), and
          - A coupled MH update using the external MH kernel.
        In addition, if the current time t is less than lag, the Y-chain is not updated (remains as in current_state).

        Args:
            current_state: Tuple (X, Y) of current states, each of shape [batch, d].
            previous_kernel_results: Dictionary containing previous kernel information (including "t" and "meeting_time").

        Returns:
            A tuple (new_state, new_kernel_results) with the updated state tuple and updated kernel results.
        """
        X, Y = current_state
        batch = tf.shape(X)[0]
        t_old = previous_kernel_results["t"]
        new_t = t_old + 1
        coin_seed = tf.stack([tf.cast(self._seed, tf.int32), new_t])
        coin = tf.random.stateless_uniform([batch], seed=coin_seed, minval=0, maxval=1, dtype=tf.float32)

        def hmc_update():
            batch = tf.shape(X)[0]
            # Sample shared momentum for both chains.
            p = self._momentum_distribution.sample(sample_shape=[batch], seed=coin_seed)
            # Simulate trajectories. Each trajectory has shape [Batch, L+1, d], logp is [Batch, L+1]
            traj_X, traj_PX, traj_logpX = simulate_trajectory(X, p, self._L, self._step_size, self._target_log_prob_fn)
            traj_Y, traj_PY, traj_logpY = simulate_trajectory(Y, p, self._L, self._step_size, self._target_log_prob_fn)
            T = tf.shape(traj_X)[0]  # T = self._L + 1
            # Compute kinetic energies as 0.5 * ||p||^2 at each trajectory point.
            kinetic_X = 0.5 * tf.reduce_sum(tf.square(traj_PX), axis=-1)
            kinetic_Y = 0.5 * tf.reduce_sum(tf.square(traj_PY), axis=-1)
            energy_X = -traj_logpX + kinetic_X  # Shape [T, batch]
            energy_Y = -traj_logpY + kinetic_Y
            # Normalize weights to probabilities along the trajectory dimension for each batch element using softmax
            norm_weights_X = tf.nn.softmax(-energy_X)
            norm_weights_Y = tf.nn.softmax(-energy_Y)
            if self._coupling_method == "maximal":
                idx1, idx2 = maximal_multinomial_coupling(norm_weights_X, norm_weights_Y)
            else:
                # TODO: For W2 coupling.
                T_float = tf.cast(T, tf.float32)
                indices = tf.cast(tf.range(T), tf.float32)
                cost_matrix = tf.square(tf.expand_dims(indices, 1) - tf.expand_dims(indices, 0))
                idx1, idx2 = w2_multinomial_coupling(norm_weights_X, norm_weights_Y, cost_matrix, reg=self._reg)

            new_X = tf.gather(traj_X, idx1, axis=1, batch_dims=1)
            new_Y = tf.gather(traj_Y, idx2, axis=1, batch_dims=1)
            return (new_X, new_Y)

        def mh_update():
            # Call the external MH kernel with the state tuple (X, Y) and corresponding bootstrap results.
            new_state, _ = self._mh_kernel.one_step(current_state, previous_kernel_results)
            return new_state

        use_mh = tf.less(coin, self._mix_prob) if self._mh_kernel is not None else tf.constant(False)
        candidate_state_mh = mh_update()
        candidate_state_hmc = hmc_update()
        candidate_X = tf.where(tf.expand_dims(use_mh, axis=-1), candidate_state_mh[0], candidate_state_hmc[0])
        candidate_Y = tf.where(tf.expand_dims(use_mh, axis=-1), candidate_state_mh[1], candidate_state_hmc[1])
        candidate_state = (candidate_X, candidate_Y)

        # For steps with time t < lag, do not update Y (keep Y unchanged).
        new_Y_final = tf.cond(tf.less(t_old, self._lag), lambda: Y, lambda: candidate_state[1])
        new_state = (candidate_state[0], new_Y_final)
        # Retrieve previous Y_history and update it.
        Y_history = previous_kernel_results["Y_history"]  # shape: [lag, batch, d]
        new_Y_history = tf.concat([Y_history[1:], tf.expand_dims(new_state[1], axis=0)], axis=0)

        # Meeting condition: for t >= lag, compare new state's X with Y_history[0] (X[t] and Y[t-lag])
        tol = tf.cast(self._tolerance, new_state[0].dtype)
        prev_mt = previous_kernel_results["meeting_time"]
        new_mt = tf.cond(
            tf.less(t_old, self._lag),
            lambda: prev_mt,
            lambda: tf.where(
                (prev_mt < 0) & (tf.reduce_max(tf.abs(new_state[0] - Y_history[0]), axis=-1) < tol),
                tf.fill(tf.shape(prev_mt), new_t),
                prev_mt,
            ),
        )
        new_kernel_results = {
            "target_log_prob": (self._target_log_prob_fn(new_state[0]), self._target_log_prob_fn(new_state[1])),
            "accepted": (tf.constant(False), tf.constant(False)),
            "proposals": (tf.zeros_like(new_state[0]), tf.zeros_like(new_state[1])),
            "current_log_prob": (self._target_log_prob_fn(new_state[0]), self._target_log_prob_fn(new_state[1])),
            "t": new_t,
            "meeting_time": new_mt,
            "Y_history": new_Y_history,
        }
        return new_state, new_kernel_results
