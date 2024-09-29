import tensorflow as tf 
import numpy as np 
import scipy.linalg

@tf.function 
def GSE_solver(A, C, D, E):
    """
    Solves the generalized Sylvester equation 
                E + AX + CXD = 0
    for X using TF.

    Args: 
        A(tf.Tensor): Tensor of shape (n+m, n+m).
        C(tf.Tensor): Tensor of shape (n+m, n+m).
        D(tf.Tensor): Tensor of shape (n,n).
        E(tf.Tensor): Tensor of shape (n+m, n).

    Returns:
        X(tf.tensor): solution of shape(n+m, n).
    """
    # Step 1: Decomposition
    AA, CC, DD, Q2, Z, F = decomposition_step(A, C, D, E)

    # Step 2: Solve uppertriangular GSE
    X = tri_GSE_solver_tf(AA, CC, DD, Q2, Z, F)

    return X

def decomposition_step(A, C, D, E):
    """
    Performs QZ and Schur decomposiotions using Scipy.
    Args:
        A(tf.Tensor): Tensor of shape (n+m, n+m).
        C(tf.Tensor): Tensor of shape (n+m, n+m).
        D(tf.Tensor): Tensor of shape (n,n).
        E(tf.Tensor): Tensor of shape (n+m, n).

    Returns 
        AA, CC, DD, Q2, Z, F(tf.tensor): Decomposed matrices as tensors.
    """

     # Use tf.py_function to wrap the Scipy decomposition
    if A.dtype == tf.float32:
        dtype = tf.complex64
        scipy_dtype = np.complex64
    elif A.dtype == tf.float64:
        dtype = tf.complex128
        scipy_dtype = np.complex128
    else:
        raise ValueError("Unsupprted dtype. Only float32 and float64 are supported.")

    def scipy_decomposition(a, c, d, e):
        
        # Perform QZ decomposition on A and C
        AA, CC, Q1, Z = scipy.linalg.qz(a, c, ouyput = 'complex')

        # Perform Schur decomposition on D
        DD, Q2 = scipy.linalg.schur(d, output='complex')

        # Calculate F
        F = Q1.conj().T @ e @ Q2

        return AA, CC, DD, Q2, Z, F

    AA, CC, DD, Q2, Z, F = tf.py_function(
        func=scipy_decomposition,
        inp=[A, C, D, E],
        Tout=[dtype, dtype, dtype,
              dtype, dtype, dtype]
    )
    AA.set_shape(A.shape)
    CC.set_shape(C.shape)
    DD.set_shape(D.shape)
    Q2.set_shape(D.shape)
    Z.set_shape(A.shape)
    F.set_shape(E.shape)

    return AA, CC, DD, Q2, Z, F 

def tri_GSE_solver_tf(AA, CC, DD, Q2, Z, F):
    """
    Solves the upper triangular Sylvester equation using TensorFlow
    Args:
        AA, CC, DD, Q2, Z, F(tf.tensor)
    Returns:
        X(tf.tensor): solution of shape(n+m, n).
    """
    N, n = F.shape

    # Initialize Y as zeros
    Y = tf.zeros_like(F, dtype=F.dtype)

    # Iterate over each column
    for k in range(n):
        # Compute the rhs: F[:, k] + CC @ Y[:, 0:k] @ DD[0:k, k]
        if k == 0:
            rhs = F[:, k]
        else:
            Y_partial = Y[:, :k] # shape (N, k)
            DD_partial = DD[:k, k] # shape (k,)
            # Compute Y_partial @ DD_partial
            term = tf.matmul(Y_partial, tf.expand_dims(DD_partial, axis=1))
            term = tf.squeeze(term, axis=1) # shape  (N,)
            rhs = F[:, k] + tf.linalg.matvec(CC, term)

        coef = AA + DD[k, k] * CC # shape (N, N)
        # Solve coef * x = rhs using least squares
        rhs = tf.expand_dims(rhs, axis=1) # shape (N, 1)
        x = tf.linalg.lstsq(coef, rhs, fast=False)[:, 0]#

        # Assign Y[:, k] = -x
        Y = tf.tensor_scatter_nd_update(Y, [[k]], [-x])
    # Compute X = Z @ Y @ Q2^H
    X = tf.matmul(tf.matmul(Z, Y), tf.linalg.adjoint(Q2))
    
    return tf.math.real(X)
