import tensorflow as tf 
import numpy as np 
import scipy.linalg

class GSE(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function 
    def GSE_solver1(self, A, C, D, E):
        """
        Solves the generalized Sylvester equation 
                    E + AX + CXD = 0
        for X using TF.

        Args: 
            A(tf.Tensor): Tensor of shape (b, n, n).
            C(tf.Tensor): Tensor of shape (b, n, n).
            D(tf.Tensor): Tensor of shape (b, n, n).
            E(tf.Tensor): Tensor of shape (b, n, n).

        Returns:
            X(tf.tensor): solution of shape(b, n, n).
        """
        # Step 1: Decomposition
        AA, CC, DD, Q2, Z, F = self.decomposition_step(A, C, D, E)

        # Step 2: Solve uppertriangular GSE
        X = self.tri_GSE_solver_tf(AA, CC, DD, Q2, Z, F)

        return X

    @tf.function
    def decomposition_step(self, A, C, D, E):
        """
        Performs QZ and Schur decomposiotions using Scipy.
            Q1* A Z = AA, Q1* C Z = CC, Q2* D Q2 = DD 
        Args:
            A, C, D, E(tf.Tensor): Tensor of shape (b, n, n)

        Returns 
            AA, CC, DD, Q2, Z, F(tf.tensor): Decomposed matrices as tensors.
            where F = Q1* E Q2
        """

        # save the input dtype
        if A.dtype == tf.float32:
            dtype = tf.complex64
            scipy_dtype = np.complex64
        elif A.dtype == tf.float64:
            dtype = tf.complex128
            scipy_dtype = np.complex128
        else:
            raise ValueError("Unsupprted dtype. Only float32 and float64 are supported.")

        def scipy_decomposition(a, c, d, e):
            # transform tensor to numpy array
            a = np.asarray(a).astype(scipy_dtype)
            c = np.asarray(c).astype(scipy_dtype)
            d = np.asarray(d).astype(scipy_dtype)
            e = np.asarray(e).astype(scipy_dtype)
            # Perform QZ decomposition on A and C
            AA, CC, Q1, Z = scipy.linalg.qz(a, c, output = 'complex')

            # Perform Schur decomposition on D
            DD, Q2 = scipy.linalg.schur(d, output='complex')

            # Calculate F
            F = Q1.conj().T @ e @ Q2

            return AA, CC, DD, Q2, Z, F

        # Use tf.map_fn to deal with batch data, tf.py_function to wrap the Scipy decomposition
        AA, CC, DD, Q2, Z, F = tf.map_fn(
            lambda x: tf.py_function(
                func=scipy_decomposition,
                inp=x,
                Tout=[dtype, dtype, dtype, dtype, dtype, dtype]
            ),
            (A, C, D, E),
            dtype=[dtype, dtype, dtype, dtype, dtype, dtype]
        )
        # define the shapes
        AA.set_shape(A.shape)
        CC.set_shape(C.shape)
        DD.set_shape(D.shape)
        Q2.set_shape(D.shape)
        Z.set_shape(A.shape)
        F.set_shape(E.shape)

        return AA, CC, DD, Q2, Z, F 

    @tf.function
    def tri_GSE_solver_tf(self, AA, CC, DD, Q2, Z, F):
        """
        Solves the upper triangular Sylvester equation using TF
                             F + AA Y + CC Y DD = 0
        Args:
            AA, CC, DD, Q2, Z, F(tf.tensor) of shape (b, n, n)
        Returns:
            X(tf.tensor): solution of shape(b, n, n).
        """
        b, n, _ = F.shape

        # Initialize tensorarray for Y
        Y = tf.TensorArray(dtype = F.dtype, size = n)

        # Iterate over each column
        for k in range(n):
            # Compute the rhs: F[:, :, k] + CC @ Y[:, :, 0:k] @ DD[:, 0:k, k]
            if k == 0:
                rhs = tf.expand_dims(F[:, :, k], axis=-1) # shape(b, n, 1)
            else:
                Y_partial = tf.transpose(Y.stack(), perm=[1,2,0])[:, :, :k] # shape (b, n, k)
                DD_partial = tf.expand_dims(DD[:, :k, k], axis=-1) # shape (b, k, 1)
                # Compute CC @ Y_partial @ DD_partial
                term = tf.matmul(CC, Y_partial) # shape (b, n, k)
                rhs = tf.expand_dims(F[:, :, k], axis=-1) + tf.matmul(term, DD_partial) # shape (b, n, 1)

            coef = AA + tf.expand_dims(tf.expand_dims(DD[:, k, k], axis=-1), axis=-1) * CC # shape (b, n, n)
            # Solve coef * x = rhs using least squares
            x = tf.linalg.lstsq(coef, rhs, fast=False)[:, :, 0] # shape (b, n)

            # Assign Y[:, :, k] = -x
            Y = Y.write(k, -x)

        # Compute X = Z @ Y @ Q2^H
        Y = Y.stack() # shape (n, b, n)
        Y = tf.transpose(Y, perm=[1,2,0]) # shape (b, n, n)
        X = tf.matmul(tf.matmul(Z, Y), tf.linalg.adjoint(Q2))
        
        return tf.math.real(X)
