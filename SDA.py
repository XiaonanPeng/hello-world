import tensorflow as tf

class SDA:
    """
    This class implements the Structure-preserving Doubling Algorithm (SDA)
    for solving the quadratic matrix equation A X^2 + B X + C = 0.
    
    The SDA class allows solving the equation using two algorithms, SDA1 and SDA2.
    
    - SDA1 requires an initial guess P0, and solves the equation iteratively (algorithm 4). 
      If P0 is not provided, a default P0 is computed using a specialized algorithm (algorithm 6).

    - SDA2 does not require an initial guess, but may encounter singularity issues (algorithm 3).
      In such cases, SDA1 is preferred or an appropriate initial guess P0 should be provided.
    
    The class provides options for setting the stopping criterion, tolerances,
    and other parameters.

    The stopping criteria options are:
    - 1: Relative change criterion
    - 2: Kahan's stopping criterion
    - 3: Residual norm criterion
    - 4: Combination of Kahan's criterion and residual norm (both must be satisfied)

    The matrix norm used is the Frobenius norm.
    
    Attributes:
        tol: Tolerance for relative change criterion.
        tol_kahan: Tolerance for Kahan's criterion (if None, defaults to tol).
        tol_residual: Tolerance for residual norm criterion (if None, defaults to tol).
        algorithm_choice_index: Choice of algorithm (1 for SDA1, 2 for SDA2).
        criterion_choice_index: Choice of stopping criterion (1, 2, 3, or 4). 
        max_iter: The maximum number of iterations in SDA to prevent infinite loops.
        rho: The upper bound of spectral radius of default P0 computed by algorithm 6, should be less than 1.
    """
    def __init__(self, tol=1e-6, tol_kahan=None, tol_residual=None,
                 algorithm_choice_index=1, criterion_choice_index=1,
                 max_iter=100, rho=0.9):
        """
        Initialize the SDA solver.
        """
        self.tol = tol
        self.tol_kahan = tol_kahan if tol_kahan is not None else tol
        self.tol_residual = tol_residual if tol_residual is not None else tol
        self.algorithm_choice_index = algorithm_choice_index
        self.criterion_choice_index = criterion_choice_index
        self.max_iter = max_iter
        self.rho = rho

    @staticmethod
    def solve_cubic(a, b, c, d):
        """
        Solve the cubic equation 
                                a x^3 + b x^2 + c x + d = 0
        using Cardono method

        Args:
            a, b, c, d: Tensor with shape ()

        Returns:
            real roots
        """
        # Calculate the intermediate values for the cubic equation
        p = (3*a*c - b**2) / (3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
        delta = (q**2 / 4 + p**3 / 27)

        # Function to compute one real root when delta > 0
        def compute_one_real_root():
            u = tf.pow(-q/2 + tf.sqrt(delta), 1/3)
            v = tf.pow(-q/2 - tf.sqrt(delta), 1/3)
            x1 = u + v - b / (3*a)
            return tf.stack([x1])

        # Function to compute three real roots when delta < 0
        def compute_three_real_roots():
            phi = tf.acos(-q / (2 * tf.sqrt(-(p**3) / 27)))
            root1 = 2 * tf.sqrt(-p / 3) * tf.cos(phi / 3) - b / (3*a)
            root2 = 2 * tf.sqrt(-p / 3) * tf.cos((phi + 2 * 3.141592653589793) / 3) - b / (3*a)
            root3 = 2 * tf.sqrt(-p / 3) * tf.cos((phi + 4 * 3.141592653589793) / 3) - b / (3*a)
            return tf.stack([root1, root2, root3])

        # Function to compute two real roots when delta == 0
        def compute_double_root():
            u = tf.pow(-q/2, 1/3)
            x1 = 2*u - b / (3*a)
            x2 = -u - b / (3*a)
            return tf.stack([x1, x2])

        # Choose the correct root computation based on delta
        roots = tf.cond(delta > 0, compute_one_real_root, 
                        lambda: tf.cond(delta < 0, compute_three_real_roots, compute_double_root))
        return roots

    def compute_default_P0(self, A, B, C):
        # Process each batch of matrices A, B, C
        def process_batch(batch):
            A, B, C = batch
            n = tf.shape(A)[0]

            # Compute optimal p_j for each column j
            def compute_pj(j):
                a_j = A[:, j]
                b_j = B[:, j]
                c_j = C[:, j]

                # Compute the coefficients for the cubic equation
                a_bar = tf.reduce_sum(tf.square(a_j))
                b_bar = 2 * tf.reduce_sum(a_j * b_j)
                c_bar = tf.reduce_sum(tf.square(b_j)) + 2 * tf.reduce_sum(a_j * c_j)
                d_bar = 2 * tf.reduce_sum(b_j * c_j)
                e_bar = tf.reduce_sum(tf.square(c_j))

                # Objective function to minimize
                def objective(p_j):
                    return a_bar * p_j**4 + b_bar * p_j**3 + c_bar * p_j**2 + d_bar * p_j + e_bar

                # Solve for roots of the cubic equation(derative of the objective function)
                roots = self.solve_cubic(4*a_bar, 3*b_bar, 2*c_bar, d_bar)

                # Evaluate objective at roots and boundaries
                candidates = tf.concat([roots, [-self.rho, self.rho]], axis=0)
                values = tf.map_fn(objective, candidates)
                min_index = tf.argmin(values)
                return candidates[min_index]

            # Compute diagonal matrix P_0
            p_j_star = tf.map_fn(compute_pj, tf.range(n), dtype=tf.float32)
            return tf.linalg.diag(p_j_star)

        # Process each batch
        P_0 = tf.map_fn(process_batch, (A, B, C), dtype=tf.float32)
        return P_0

    def stopping_criterion(self, Xk, Xk_prev, Xk_prev_prev=None, Rk=None):
        """
        Compute the stopping criterion based on the criterion_choice_index.

        Args:
            Xk: Current iterate X_k
            Xk_prev: Previous iterate X_{k-1}
            Xk_prev_prev: Previous previous iterate X_{k-2} (only needed for Kahan's criterion)
            Rk: Residual at current iterate (only needed for residual norm criterion)

        Returns:
            stop: Boolean indicating whether to stop
        """
        # Compute norms using Frobenius norm

        if self.criterion_choice_index == 1:
            # Relative change criterion
            lhs = tf.norm(Xk - Xk_prev, axis=[-2,-1])
            rhs = self.tol * tf.norm(Xk, axis=[-2,-1])
            criterion = lhs <= rhs

        elif self.criterion_choice_index == 2:
            # Kahan's criterion
            norm_Xk_Xkprev = tf.norm(Xk - Xk_prev, axis=[-2,-1])
            norm_Xkprev_Xkprevprev = tf.norm(Xk_prev - Xk_prev_prev, axis=[-2,-1])
            lhs = norm_Xk_Xkprev ** 2
            rhs = self.tol_kahan * tf.norm(Xk, axis=[-2,-1]) * (norm_Xkprev_Xkprevprev - norm_Xk_Xkprev)
            criterion = lhs <= rhs

        elif self.criterion_choice_index == 3:
            # Residual norm criterion
            criterion = tf.norm(Rk, axis=[-2,-1]) <= self.tol_residual

        elif self.criterion_choice_index == 4:
            # Combination of criterion 2 and criterion 3
            # Both Kahan's criterion and residual norm must be satisfied
            # Kahan's criterion
            norm_Xk_Xkprev = tf.norm(Xk - Xk_prev, axis=[-2,-1])
            norm_Xkprev_Xkprevprev = tf.norm(Xk_prev - Xk_prev_prev, axis=[-2,-1])
            lhs = norm_Xk_Xkprev ** 2
            rhs = self.tol_kahan * tf.norm(Xk, axis=[-2,-1]) * (norm_Xkprev_Xkprevprev - norm_Xk_Xkprev)
            kahan_criterion = lhs <= rhs
            # Residual norm criterion
            residual_criterion = tf.norm(Rk, axis=[-2,-1]) <= self.tol_residual
            criterion = tf.logical_and(kahan_criterion, residual_criterion)
        else:
            raise ValueError("Invalid criterion choice index.")

        # Return the looping condition
        return tf.logical_not(criterion)

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
    
        # Implement SDA1
        if self.algorithm_choice_index == 1:
            if P0 is None:
                P0 = self.compute_default_P0(A, B, C)  

            solution, ite = tf.map_fn(
                self.sda1_per_batch,
                elems = (A, B, C, P0),
                fn_output_signature = (tf.TensorSpec(shape=A.shape[1:], dtype=A.dtype), tf.TensorSpec(shape=(), dtype=tf.int32)),
                parallel_iterations = None
            )
        # Implement SDA2
        elif self.algorithm_choice_index == 2:
            solution, ite = tf.map_fn(
                self.sda2_per_batch,
                elems=(A, B, C),
                fn_output_signature = (tf.TensorSpec(shape=A.shape[1:], dtype=A.dtype), tf.TensorSpec(shape=(), dtype=tf.int32)),
                parallel_iterations = None
            )
        else:
            raise ValueError("Invalid algorithm choice index.")
        return solution, ite


    def sda1_per_batch(self, inputs):
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
        A, B, C, P0 = inputs
        n = tf.shape(A)[-1]
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

        def cond(X, Y, E, F, X_prev, X_prev_prev, k):
            # compute the redusial matrix Rk
            Pk = X + P0
            Pk_square = tf.matmul(Pk, Pk)
            Rk = tf.matmul(A, Pk_square) + tf.matmul(B, Pk) + C
            # make sure the there are at least 3 iterations and at most max_iter iterations
            return tf.logical_and(k < self.max_iter, tf.logical_or(k < 3, self.stopping_criterion(X, X_prev, X_prev_prev, Rk)))


        def body(X, Y, E, F, X_prev, X_prev_prev, k):
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

            return X_new, Y_new, E_new, F_new, X_prev, X_prev_prev, k+1

        X_final, _, _, _, _, _, T = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[X, Y, E, F, X_prev, X_prev_prev, k]
        )

        Pk = X_final + P0
        return Pk, T

    def sda2_per_batch(self, inputs):
        """
        Solve the quadratic matrix equation using SDA2 algorithm for a single batch.

        Args:
            A: Tensor of shape (n, n)
            B: Tensor of shape (n, n)
            C: Tensor of shape (n, n)

        Returns:
            Pk: Solution matrix of shape (n, n)
        """
        A, B, C= inputs
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

        def cond(X, Y, E, F, X_prev, X_prev_prev, k):
            # compute the residual matirx Rk
            X_B = X + B
            inv_X_B = tf.linalg.inv(X_B)
            Pk = -tf.matmul(inv_X_B, C)
            Pk_square = tf.matmul(Pk, Pk)
            Rk = tf.matmul(A, Pk_square) + tf.matmul(B, Pk) + C
            # make sure the there are at least 3 iterations and at most max_iter iterations
            return tf.logical_and(k < self.max_iter, tf.logical_or(k < 3, self.stopping_criterion(X, X_prev, X_prev_prev, Rk)))


        def body(X, Y, E, F, X_prev, X_prev_prev, k):
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

            return X_new, Y_new, E_new, F_new, X_prev, X_prev_prev, k+1

        X_final, _, _, _, _, _, T = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[X, Y, E, F, X_prev, X_prev_prev, k]
        )

        X_B = X_final + B
        inv_X_B = tf.linalg.inv(X_B)
        Pk = -tf.matmul(inv_X_B, C)
        return Pk, T
