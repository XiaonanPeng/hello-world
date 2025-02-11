#!/usr/bin/env python
"""
coupling.py

This module implements coupling algorithms for multinomial distributions 
to be used in coupled multinomial HMC. In particular, it provides:

1. maximal_multinomial_coupling: 
   Given two categorical distributions (represented as probability vectors), 
   it returns a pair of indices from a maximal coupling.

2. w2_multinomial_coupling:
   Given two categorical distributions and a cost (distance) matrix,
   it returns a pair of indices from an approximate optimal transport (W2) coupling 
   using the Sinkhorn algorithm.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def maximal_multinomial_coupling(mu, nu):
    """
    Performs maximal coupling of two categorical distributions.
    
    Args:
        mu: Tensor of shape [K] representing the probability vector of the first distribution.
        nu: Tensor of shape [K] representing the probability vector of the second distribution.
        
    Returns:
        A tuple of two integers (i, j) representing the coupled indices.
    """
    mu = tf.convert_to_tensor(mu, dtype=tf.float32)
    nu = tf.convert_to_tensor(nu, dtype=tf.float32)
    
    # Compute element-wise minimum and its sum (omega)
    min_vals = tf.minimum(mu, nu)
    omega = tf.reduce_sum(min_vals)
    
    # Sample a uniform random number.
    u = tf.random.uniform([], dtype=tf.float32)
    
    def coupled_same():
        # Sample index from categorical distribution with probabilities proportional to min_vals.
        cat = tfp.distributions.Categorical(probs=min_vals)
        i = cat.sample()
        # Set j equal to i.
        return (i, i)
    
    def coupled_different():
        # Compute residual probabilities for mu and nu.
        res_mu = mu - min_vals
        res_nu = nu - min_vals
        sum_res = 1.0 - omega
        # Normalize residual probabilities.
        probs_mu = res_mu / sum_res
        probs_nu = res_nu / sum_res
        cat_mu = tfp.distributions.Categorical(probs=probs_mu)
        cat_nu = tfp.distributions.Categorical(probs=probs_nu)
        i = cat_mu.sample()
        j = cat_nu.sample()
        return (i, j)
    
    return tf.cond(u < omega, coupled_same, coupled_different)


def w2_multinomial_coupling(mu, nu, cost, reg=1e-1, num_iters=100):
    """
    Performs an approximate optimal transport (W2) coupling using the Sinkhorn algorithm.
    
    Args:
        mu: Tensor of shape [K] representing the probability vector of the first distribution.
        nu: Tensor of shape [K] representing the probability vector of the second distribution.
        cost: Tensor of shape [K, K] representing the cost (distance) matrix.
        reg: Entropic regularization parameter.
        num_iters: Number of iterations for the Sinkhorn algorithm.
        
    Returns:
        A tuple of two integers (i, j) representing the coupled indices.
    """
    mu = tf.convert_to_tensor(mu, dtype=tf.float32)
    nu = tf.convert_to_tensor(nu, dtype=tf.float32)
    cost = tf.convert_to_tensor(cost, dtype=tf.float32)
    K_mat = tf.exp(-cost / reg)  # shape [K, K]
    
    # Initialize scaling factors.
    u = tf.ones_like(mu, dtype=tf.float32)
    v = tf.ones_like(nu, dtype=tf.float32)
    
    # Sinkhorn iterations.
    for _ in range(num_iters):
        u = mu / (tf.linalg.matvec(K_mat, v) + 1e-8)
        v = nu / (tf.linalg.matvec(tf.transpose(K_mat), u) + 1e-8)
    
    # Compute the coupling matrix.
    coupling_matrix = tf.linalg.diag(u) @ K_mat @ tf.linalg.diag(v)
    coupling_matrix = coupling_matrix / tf.reduce_sum(coupling_matrix)
    
    # Flatten the coupling matrix and sample an index.
    flat_coupling = tf.reshape(coupling_matrix, [-1])
    cat = tfp.distributions.Categorical(probs=flat_coupling)
    index = cat.sample()
    K_size = tf.shape(mu)[0]
    i = index // K_size
    j = index % K_size
    return (i, j)
