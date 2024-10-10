import tensorflow as tf
import numpy as np

class SDA:
    """
    This class implements the Structure-preserving Doubling Algorithm (SDA)
    for solving the quadratic matrix equation A X^2 + B X + C = 0.

    The SDA class allows solving the equation using two algorithms, SDA1 and SDA2.

    - **SDA1** requires an initial guess P0, and solves the equation iteratively.
      If P0 is not provided, a default P0 is computed using a specialized algorithm
      (by finding the minimizer of a quartic function for each diagonal entry).
      SDA1 is generally more stable and preferred when SDA2 encounters singularities.

    - **SDA2** does not require an initial guess, but may encounter singularity issues.
      In such cases, SDA1 is preferred or an appropriate initial guess P0 should be provided.

    The class provides options for setting the stopping criterion, tolerances,
    and other parameters.

    The stopping criteria options are:
    - **1**: Relative change criterion
    - **2**: Kahan's stopping criterion
    - **3**: Residual norm criterion
    - **4**: Combination of Kahan's criterion and residual norm (both must be satisfied)

    The matrix norm used is the Frobenius norm.

    Attributes:
        tol: Tolerance for relative change criterion.
        tol_kahan: Tolerance for Kahan's criterion.
        tol_residual: Tolerance for residual norm criterion.
        algorithm_choice_index: Choice of algorithm (1 for SDA1, 2 for SDA2).
        criterion_choice_index: Choice of stopping criterion (1, 2, 3, or 4).
        max_iter: Maximum number of iterations to prevent infinite loops.
    """
    def __init__(self,
                 tol=1e-6,
                 algorithm_choice_index=1,
                 criterion_choice_index=1,
                 tol_kahan=None,
                 tol_residual=None,
                 max_iter=1000):
        """
        Initialize the SDA solver.

        Args:
            tol: Tolerance for relative change criterion.
            algorithm_choice_index: Algorithm choice index (1 for SDA1, 2 for SDA2).
            criterion_choice_index: Criterion choice index:
                1 for relative change,
                2 for Kahan's criterion,
                3 for residual norm,
                4 for both Kahan's criterion and residual norm (both must be satisfied).
            tol_kahan: Tolerance for Kahan's criterion (if None, defaults to tol).
            tol_residual: Tolerance for residual norm criterion (if None, defaults to tol).
            max_iter: Maximum number of iterations to prevent infinite loops.
        """
        self.tol = tol
        self.algorithm_choice_index = algorithm_choice_index
        self.criterion_choice_index = criterion_choice_index
        self.tol_kahan = tol_kahan if tol_kahan is not None else tol
        self.tol_residual = tol_residual if tol_residual is not None else tol
        self.max_iter = max_iter

    @staticmethod
    def cbrt(x):
        """
        Compute the cube root of x, handling negative values.

        Args:
            x: Tensor

        Returns:
            Cube root of x.
        """
        return tf.sign(x) * tf.pow(tf.abs(x), 1.0 / 3.0)

    def compute_default_P0(self, A, B, C):
        """
        Compute default initial guess P0 for SDA1 when P0 is not provided.

        Returns:
            P0: Tensor of shape (batch_size, n, n)
        """
        def compute_P0_per_batch(inputs):
            Ai, Bi, Ci = inputs
            n = tf.shape(Ai)[0]

            # Extract diagonals of Ai, Bi, Ci
            a_diag = tf.linalg.diag_part(Ai)
            b_diag = tf.linalg.diag_part(Bi)
            c_diag = tf.linalg.diag_part(Ci)

            # Compute tilde coefficients for diagonals
            tilde_a = tf.square(a_diag)
            tilde_b = 2 * a_diag * b_diag
            tilde_c = tf.square(b_diag) + 2 * a_diag * c_diag
            tilde_d = 2 * b_diag * c_diag
            tilde_e = tf.square(c_diag)

            # Compute derivative coefficients
            a_C = 4 * tilde_a
            b_C = 3 * tilde_b
            c_C = 2 * tilde_c
            d_C = tilde_d

            # Initialize p_list
            p_list = []

            for j in tf.range(n):
                # For each diagonal entry
                aj = tilde_a[j]
                bj = tilde_b[j]
                cj = tilde_c[j]
                dj = tilde_d[j]
                ej = tilde_e[j]

                aC = a_C[j]
                bC = b_C[j]
                cC = c_C[j]
                dC = d_C[j]

                roots = []

                if tf.abs(aC) > 1e-12:
                    # Normalize coefficients
                    b_N = bC / aC
                    c_N = cC / aC
                    d_N = dC / aC

                    p_coef = c_N - b_N ** 2 / 3
                    q_coef = (2 * b_N ** 3) / 27 - (b_N * c_N) / 3 + d_N

                    delta = (q_coef / 2) ** 2 + (p_coef / 3) ** 3

                    rho = 0.9

                    if delta > 0:
                        sqrt_delta = tf.sqrt(delta)
                        u = self.cbrt(-q_coef / 2 + sqrt_delta)
                        v = self.cbrt(-q_coef / 2 - sqrt_delta)
                        t = u + v
                        x = t - b_N / 3
                        roots.append(x)
                    elif tf.abs(delta) < 1e-12:
                        u = self.cbrt(-q_coef / 2)
                        t1 = 2 * u
                        t2 = -u
                        x1 = t1 - b_N / 3
                        x2 = t2 - b_N / 3
                        roots.extend([x1, x2])
                    else:
                        phi = tf.acos(-q_coef / (2 * tf.sqrt(-(p_coef / 3) ** 3)))
                        r = 2 * tf.sqrt(-p_coef / 3)
                        x1 = r * tf.cos(phi / 3) - b_N / 3
                        x2 = r * tf.cos((phi + 2 * np.pi) / 3) - b_N / 3
                        x3 = r * tf.cos((phi + 4 * np.pi) / 3) - b_N / 3
                        roots.extend([x1, x2, x3])
                else:
                    # Quadratic case
                    discriminant = cC ** 2 - 4 * bC * dC
                    if discriminant >= 0:
                        sqrt_disc = tf.sqrt(discriminant)
                        x1 = (-cC + sqrt_disc) / (2 * bC)
                        x2 = (-cC - sqrt_disc) / (2 * bC)
                        roots.extend([x1, x2])

                # Add -rho and rho
                rho = 0.9
                roots.extend([-rho, rho])

                # Convert roots to tensor
                roots = tf.stack(roots)
                roots = tf.boolean_mask(roots, tf.math.is_finite(roots))

                # Clip roots to [-rho, rho]
                roots_clipped = tf.clip_by_value(roots, -rho, rho)

                # Remove duplicates
                roots_unique = tf.unique(roots_clipped).y

                # Evaluate r_j(p_j)
                p_powers = [tf.pow(roots_unique, k) for k in range(5)]  # k from 0 to 4
                r_p = aj * p_powers[4] + bj * p_powers[3] + cj * p_powers[2] + dj * p_powers[1] + ej

                # Find p_j minimizing r_j(p_j)
                min_index = tf.argmin(r_p)
                p_j = roots_unique[min_index]
                p_list.append(p_j)

            # Stack p_list to form diagonal matrix P0_i
            p_list_tensor = tf.stack(p_list)
            P0_i = tf.linalg.diag(p_list_tensor)
            return P0_i

        P0 = tf.map_fn(
            compute_P0_per_batch,
            elems=(A, B, C),
            dtype=A.dtype
        )
        return P0

    def solve(self, A, B, C, P0=None):
        """
        Solve the quadratic matrix equation using the selected SDA algorithm.

        Args:
            A: Tensor of shape (b, n, n)
            B: Tensor of shape (b, n, n)
            C: Tensor of shape (b, n, n)
            P0: Tensor of shape (b, n, n), initial guess for SDA1

        Returns:
            Solution matrix P_k of shape (b, n, n)
        """
        if self.algorithm_choice_index == 1:
            if P0 is None:
                P0 = self.compute_default_P0(A, B, C)
            solution = tf.map_fn(
                lambda inputs: self.sda1_per_batch(*inputs),
                elems=(A, B, C, P0),
                dtype=A.dtype
            )
        elif self.algorithm_choice_index == 2:
            solution = tf.map_fn(
                lambda inputs: self.sda2_per_batch(*inputs),
                elems=(A, B, C),
                dtype=A.dtype
            )
        else:
            raise ValueError("Invalid algorithm choice index.")
        return solution

    def sda1_per_batch(self, A, B, C, P0):
        """
        Solve the quadratic matrix equation using SDA1 algorithm for a single batch.

        Args:
            A: Tensor of shape (n, n)
            B: Tensor of shape (n, n)
            C: Tensor of shape (n, n)
            P0: Tensor of shape (n, n)

        Returns:
            Pk: Solution matrix of shape (n, n)
        """
        n = tf.shape(A)[0]
        I = tf.eye(n, dtype=A.dtype)

        # Compute initial variables
        AP0 = tf.matmul(A, P0)
        B_AP0 = B + AP0

        # Inversion of B_AP0
        inv_B_AP0 = tf.linalg.inv(B_AP0)

        X = -P0 - tf.matmul(inv_B_AP0, C)
        Y = -tf.matmul(inv_B_AP0, A)
        E = -tf.matmul(inv_B_AP0, C)
        F = -tf.matmul(inv_B_AP0, A)

        X_prev = X
        X_prev_prev = X
        k = 0

        # Stopping condition variables
        converged = False

        def cond(args):
            X, Y, E, F, X_prev, X_prev_prev, k = args
            return tf.logical_and(tf.less(k, self.max_iter), tf.logical_not(converged))

        def body(args):
            X, Y, E, F, X_prev, X_prev_prev, k = args

            Pk = X + P0
            Pk_square = tf.matmul(Pk, Pk)
            Rk = tf.matmul(A, Pk_square) + tf.matmul(B, Pk) + C

            # Compute stop condition
            stop = self.stopping_criterion_single(X, X_prev, X_prev_prev, Rk)

            # Update converged status
            converged = stop

            X_prev_prev = X_prev
            X_prev = X

            # Compute (I - Y X) and (I - X Y)
            I_minus_YX = I - tf.matmul(Y, X)
            I_minus_XY = I - tf.matmul(X, Y)

            # Compute inverses
            inv_I_minus_YX = tf.linalg.inv(I_minus_YX)
            inv_I_minus_XY = tf.linalg.inv(I_minus_XY)

            # Update E, F, X, Y
            E_new = tf.matmul(tf.matmul(E, inv_I_minus_YX), E)
            F_new = tf.matmul(tf.matmul(F, inv_I_minus_XY), F)
            X_new = X + tf.matmul(tf.matmul(F, inv_I_minus_XY), tf.matmul(X, E))
            Y_new = Y + tf.matmul(tf.matmul(E, inv_I_minus_YX), tf.matmul(Y, F))

            return [X_new, Y_new, E_new, F_new, X_prev, X_prev_prev, k+1]

        [X_final, _, _, _, _, _, _] = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[X, Y, E, F, X_prev, X_prev_prev, k]
        )

        Pk = X_final + P0
        return Pk

    def sda2_per_batch(self, A, B, C):
        """
        Solve the quadratic matrix equation using SDA2 algorithm for a single batch.

        Args:
            A: Tensor of shape (n, n)
            B: Tensor of shape (n, n)
            C: Tensor of shape (n, n)

        Returns:
            Pk: Solution matrix of shape (n, n)
        """
        n = tf.shape(A)[0]
        I = tf.eye(n, dtype=A.dtype)

        # Initialize variables
        X = tf.zeros_like(A)
        Y = -B
        E = -C
        F = -A

        X_prev = X
        X_prev_prev = X
        k = 0

        # Stopping condition variables
        converged = False

        def cond(args):
            X, Y, E, F, X_prev, X_prev_prev, k = args
            return tf.logical_and(tf.less(k, self.max_iter), tf.logical_not(converged))

        def body(args):
            X, Y, E, F, X_prev, X_prev_prev, k = args

            X_B = X + B
            inv_X_B = tf.linalg.inv(X_B)
            Pk = -tf.matmul(inv_X_B, C)
            Pk_square = tf.matmul(Pk, Pk)
            Rk = tf.matmul(A, Pk_square) + tf.matmul(B, Pk) + C

            # Compute stop condition
            stop = self.stopping_criterion_single(X, X_prev, X_prev_prev, Rk)

            # Update converged status
            converged = stop

            X_prev_prev = X_prev
            X_prev = X

            # Compute (X - Y)
            X_minus_Y = X - Y
            inv_X_minus_Y = tf.linalg.inv(X_minus_Y)

            # Update E, F, X, Y
            E_new = tf.matmul(tf.matmul(E, inv_X_minus_Y), E)
            F_new = tf.matmul(tf.matmul(F, inv_X_minus_Y), F)
            X_new = X - tf.matmul(tf.matmul(F, inv_X_minus_Y), E)
            Y_new = Y + tf.matmul(tf.matmul(E, inv_X_minus_Y), F)

            return [X_new, Y_new, E_new, F_new, X_prev, X_prev_prev, k+1]

        [X_final, _, _, _, _, _, _] = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[X, Y, E, F, X_prev, X_prev_prev, k]
        )

        X_B = X_final + B
        inv_X_B = tf.linalg.inv(X_B)
        Pk = -tf.matmul(inv_X_B, C)
        return Pk

    def stopping_criterion_single(self, Xk, Xk_prev, Xk_prev_prev=None, Rk=None):
        """
        Compute the stopping criterion for a single batch element.

        Args:
            Xk: Current iterate X_k
            Xk_prev: Previous iterate X_{k-1}
            Xk_prev_prev: Previous previous iterate X_{k-2} (only needed for Kahan's criterion)
            Rk: Residual at current iterate (only needed for residual norm criterion)

        Returns:
            stop: Boolean indicating whether to stop
        """
        if self.criterion_choice_index == 1:
            # Relative change criterion
            numerator = tf.linalg.norm(Xk - Xk_prev, ord='fro')
            denominator = tf.linalg.norm(Xk, ord='fro')
            criterion = numerator <= self.tol * denominator

        elif self.criterion_choice_index == 2:
            # Kahan's criterion
            if Xk_prev_prev is None:
                criterion = False
            else:
                norm_Xk_Xkprev = tf.linalg.norm(Xk - Xk_prev, ord='fro')
                norm_Xkprev_Xkprevprev = tf.linalg.norm(Xk_prev - Xk_prev_prev, ord='fro')
                numerator = norm_Xk_Xkprev ** 2
                denominator = norm_Xkprev_Xkprevprev - norm_Xk_Xkprev
                denominator = tf.where(tf.abs(denominator) < 1e-12, 1e-12, denominator)
                lhs = numerator / denominator
                rhs = self.tol_kahan * tf.linalg.norm(Xk, ord='fro')
                criterion = lhs <= rhs

        elif self.criterion_choice_index == 3:
            # Residual norm criterion
            criterion = tf.linalg.norm(Rk, ord='fro') <= self.tol_residual

        elif self.criterion_choice_index == 4:
            # Combination of Kahan's criterion and residual norm
            if Xk_prev_prev is None:
                kahan_criterion = False
            else:
                norm_Xk_Xkprev = tf.linalg.norm(Xk - Xk_prev, ord='fro')
                norm_Xkprev_Xkprevprev = tf.linalg.norm(Xk_prev - Xk_prev_prev, ord='fro')
                numerator = norm_Xk_Xkprev ** 2
                denominator = norm_Xkprev_Xkprevprev - norm_Xk_Xkprev
                denominator = tf.where(tf.abs(denominator) < 1e-12, 1e-12, denominator)
                lhs = numerator / denominator
                rhs = self.tol_kahan * tf.linalg.norm(Xk, ord='fro')
                kahan_criterion = lhs <= rhs
            residual_criterion = tf.linalg.norm(Rk, ord='fro') <= self.tol_residual
            criterion = tf.logical_and(kahan_criterion, residual_criterion)
        else:
            raise ValueError("Invalid criterion choice index.")

        return criterion
