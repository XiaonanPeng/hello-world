
import tensorflow as tf

class SDA:
    """
    This class implements the Structure-preserving Doubling Algorithm (SDA)
    for solving the quadratic matrix equation A X^2 + B X + C = 0.
    
    The SDA class allows solving the equation using two algorithms, SDA1 and SDA2.
    
    - SDA1 requires an initial guess P0, and solves the equation iteratively.
      If P0 is not provided, a default P0 is computed using a specialized algorithm.

    - SDA2 does not require an initial guess, but may encounter singularity issues.
      In such cases, SDA1 is preferred or an appropriate initial guess P0 should be provided.
    
    The class provides options for setting the stopping criterion, tolerances,
    and other parameters.

    The stopping criteria options are:
    - 1: Relative change criterion
    - 2: Kahan's stopping criterion
    - 3: Residual norm criterion
    - 4: Combination of Kahan's criterion and residual norm (both must be satisfied)
    
    Attributes:
        tol: Tolerance for relative change criterion.
        tol_kahan: Tolerance for Kahan's criterion.
        tol_residual: Tolerance for residual norm criterion.
        algorithm_choice_index: Choice of algorithm (1 for SDA1, 2 for SDA2).
        criterion_choice_index: Choice of stopping criterion (1, 2, 3, or 4).
        A, B, C: Coefficient matrices of the quadratic equation.
        P0: Initial guess for SDA1.
    """
    def __init__(self, tol=1e-6, algorithm_choice_index=1, criterion_choice_index=1,
                 tol_kahan=None, tol_residual=None):
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
        """
        self.tol = tol
        self.algorithm_choice_index = algorithm_choice_index
        self.criterion_choice_index = criterion_choice_index
        self.tol_kahan = tol_kahan if tol_kahan is not None else tol
        self.tol_residual = tol_residual if tol_residual is not None else tol
        # Initialize matrices as None; will be set in the input() method
        self.A = None
        self.B = None
        self.C = None
        self.P0 = None

    def input(self, A, B, C, P0=None):
        """
        Input coefficient matrices A, B, C, and the initial guess P0.

        Args:
            A: Tensor of shape (b, n, n)
            B: Tensor of shape (b, n, n)
            C: Tensor of shape (b, n, n)
            P0: Tensor of shape (b, n, n), initial guess (only required for SDA1)
        """
        self.A = A
        self.B = B
        self.C = C
        self.P0 = P0

    def compute_default_P0(self):
        """
        Compute default initial guess P0 for SDA1 when P0 is not provided.

        Uses an algorithm to compute P0 as a diagonal matrix with entries p_j,
        where each p_j is computed by solving a minimization problem involving
        the coefficients of the quadratic equation.

        Returns:
            P0: Tensor of shape (b, n, n)
        """
        # Compute default P0

        # Note: Since this involves solving cubic equations, we provide a simplified version.
        # If precise roots are needed, you may implement a cubic solver or provide P0 explicitly.

        # For simplicity, we set rho as a small positive scalar less than 1
        rho = 0.9  # This can be adjusted as needed

        A = self.A
        B = self.B
        C = self.C
        batch_size = tf.shape(A)[0]
        n = tf.shape(A)[1]

        # P0 will be a batch of diagonal matrices
        P0 = []

        # We will loop over batch size
        for i in range(batch_size):
            # Extract matrices for batch element i
            Ai = A[i]
            Bi = B[i]
            Ci = C[i]

            # Initialize p_list for this batch element
            p_list = []

            # For each diagonal entry p_j
            for j in range(n):
                # For scalar case, a_j, b_j, c_j are columns of A, B, C
                a_j = Ai[:, j]
                b_j = Bi[:, j]
                c_j = Ci[:, j]

                # Compute coefficients
                tilde_b = 2 * tf.tensordot(a_j, b_j, axes=1)
                tilde_c = tf.tensordot(b_j, b_j, axes=1) + 2 * tf.tensordot(a_j, c_j, axes=1)
                tilde_d = 2 * tf.tensordot(b_j, c_j, axes=1)
                tilde_e = tf.tensordot(c_j, c_j, axes=1)

                # The quartic function reduces to a cubic since tilde_a = 0
                # Compute derivative r_j'(p_j) = 3 tilde_b p_j^2 + 2 tilde_c p_j + tilde_d = 0

                # Coefficients for quadratic equation a p^2 + b p + c = 0
                a_coeff = 3 * tilde_b
                b_coeff = 2 * tilde_c
                c_coeff = tilde_d

                # Solve a_coeff * p^2 + b_coeff * p + c_coeff = 0
                discriminant = b_coeff ** 2 - 4 * a_coeff * c_coeff

                # Collect possible p_j^(i)
                p_candidates = []

                if a_coeff == 0:
                    # Linear equation
                    if b_coeff != 0:
                        p_root = - c_coeff / b_coeff
                        p_candidates.append(p_root)
                else:
                    if discriminant >= 0:
                        sqrt_disc = tf.sqrt(discriminant)
                        p_root1 = (-b_coeff + sqrt_disc) / (2 * a_coeff)
                        p_root2 = (-b_coeff - sqrt_disc) / (2 * a_coeff)
                        p_candidates.extend([p_root1, p_root2])
                    else:
                        # Discriminant is negative, no real roots
                        pass

                # Add -rho and rho to the candidate list
                p_candidates.extend([-rho, rho])

                # For each candidate, if p_j^(i) not in [-rho, rho], set it to rho
                p_candidates = [p if -rho <= p <= rho else rho for p in p_candidates]
                # Remove duplicates
                p_candidates = list(set(p_candidates))

                # Evaluate r_j(p_j) for each candidate
                r_values = []
                for p in p_candidates:
                    # Compute r_j(p_j) = tilde_b p^3 + tilde_c p^2 + tilde_d p + tilde_e
                    r_p = tilde_b * p ** 3 + tilde_c * p ** 2 + tilde_d * p + tilde_e
                    r_values.append(r_p)

                # Find p_j minimizing r_j(p_j)
                min_index = tf.argmin(r_values)
                p_j = p_candidates[min_index]
                p_list.append(p_j)

            # Create diagonal matrix for P0_i
            P0_i = tf.linalg.diag(p_list)
            P0.append(P0_i)

        # Stack P0_i to form P0 of shape (batch_size, n, n)
        P0 = tf.stack(P0)
        return P0

    def stopping_criterion(self, Xk, Xk_prev, Xk_prev_prev=None, Rk=None):
        """
        Compute the stopping criterion based on the criterion_choice_index.

        Args:
            Xk: Current iterate X_k
            Xk_prev: Previous iterate X_{k-1}
            Xk_prev_prev: Previous previous iterate X_{k-2} (only needed for Kahan's criterion)
            Rk: Residual at current iterate (only needed for residual norm criterion)

        Returns:
            stop: Tensor of booleans indicating whether to stop per batch element
        """
        # Compute norms using Frobenius norm
        batch_size = tf.shape(Xk)[0]
        if self.criterion_choice_index == 1:
            # Relative change criterion
            numerator = tf.linalg.norm(Xk - Xk_prev, ord='fro', axis=[-2, -1])
            denominator = tf.linalg.norm(Xk, ord='fro', axis=[-2, -1])
            criterion = numerator <= self.tol * denominator

        elif self.criterion_choice_index == 2:
            # Kahan's criterion
            if Xk_prev_prev is None:
                # Can't compute Kahan's criterion without X_{k-2}
                criterion = tf.constant(False, shape=[batch_size], dtype=tf.bool)
            else:
                norm_Xk_Xkprev = tf.linalg.norm(Xk - Xk_prev, ord='fro', axis=[-2, -1])
                norm_Xkprev_Xkprevprev = tf.linalg.norm(Xk_prev - Xk_prev_prev, ord='fro', axis=[-2, -1])
                numerator = norm_Xk_Xkprev ** 2
                denominator = norm_Xkprev_Xkprevprev - norm_Xk_Xkprev
                # To avoid division by zero or negative denominator
                denominator = tf.where(tf.abs(denominator) < 1e-12, tf.ones_like(denominator) * 1e-12, denominator)
                lhs = numerator / denominator
                rhs = self.tol_kahan * tf.linalg.norm(Xk, ord='fro', axis=[-2, -1])
                criterion = lhs <= rhs

        elif self.criterion_choice_index == 3:
            # Residual norm criterion
            criterion = tf.linalg.norm(Rk, ord='fro', axis=[-2, -1]) <= self.tol_residual

        elif self.criterion_choice_index == 4:
            # Combination of criterion 2 and criterion 3
            # Both Kahan's criterion and residual norm must be satisfied
            if Xk_prev_prev is None:
                kahan_criterion = tf.constant(False, shape=[batch_size], dtype=tf.bool)
            else:
                # Kahan's criterion
                norm_Xk_Xkprev = tf.linalg.norm(Xk - Xk_prev, ord='fro', axis=[-2, -1])
                norm_Xkprev_Xkprevprev = tf.linalg.norm(Xk_prev - Xk_prev_prev, ord='fro', axis=[-2, -1])
                numerator = norm_Xk_Xkprev ** 2
                denominator = norm_Xkprev_Xkprevprev - norm_Xk_Xkprev
                denominator = tf.where(tf.abs(denominator) < 1e-12, tf.ones_like(denominator) * 1e-12, denominator)
                lhs = numerator / denominator
                rhs = self.tol_kahan * tf.linalg.norm(Xk, ord='fro', axis=[-2, -1])
                kahan_criterion = lhs <= rhs
            # Residual norm criterion
            residual_criterion = tf.linalg.norm(Rk, ord='fro', axis=[-2, -1]) <= self.tol_residual
            criterion = tf.logical_and(kahan_criterion, residual_criterion)
        else:
            raise ValueError("Invalid criterion choice index.")

        # Return the stopping status per batch element
        return criterion

    def solve(self):
        """
        Solve the quadratic matrix equation using the selected SDA algorithm.

        Returns:
            Solution matrix P_k of shape (b, n, n)
        """
        if self.algorithm_choice_index == 1:
            return self.sda1()
        elif self.algorithm_choice_index == 2:
            return self.sda2()
        else:
            raise ValueError("Invalid algorithm choice index.")

    def sda1(self):
        """
        Solve the quadratic matrix equation using SDA1 algorithm.

        Returns:
            Solution matrix P_k of shape (b, n, n)
        """
        # Algorithm SDA1 with inputs A, B, C, P0.

        A = self.A
        B = self.B
        C = self.C
        P0 = self.P0

        if P0 is None:
            # Compute default P0
            P0 = self.compute_default_P0()

        # Initialize variables
        # First, compute (B + A P0)
        batch_size = tf.shape(A)[0]
        n = tf.shape(A)[1]

        I = tf.eye(n, batch_shape=[batch_size], dtype=A.dtype)
        k = 0  # Iteration counter

        Pk = P0  # Initial guess

        # Compute B + A P0
        AP0 = tf.matmul(A, P0)
        B_AP0 = B + AP0

        # Inversion of B + A P0
        try:
            inv_B_AP0 = tf.linalg.inv(B_AP0)
        except tf.errors.InvalidArgumentError:
            raise ValueError("Matrix inversion failed in SDA1: B + A*P0 is singular. "
                             "Please provide a different initial guess P0.")

        # Initialize X0, Y0, E0, F0
        X = -P0 - tf.matmul(inv_B_AP0, C)
        Y = -tf.matmul(inv_B_AP0, A)
        E = -tf.matmul(inv_B_AP0, C)
        F = -tf.matmul(inv_B_AP0, A)

        # Initialize previous iterates
        Xk_prev = X
        Xk_prev_prev = None  # For Kahan's criterion
        Rk = None  # For residual norm criterion

        while True:
            # Compute the stopping criterion
            if self.criterion_choice_index in [3, 4]:
                # Compute residual norm: Rk = A Pk^2 + B Pk + C
                Pk = X + P0
                Pk_square = tf.matmul(Pk, Pk)
                Rk = tf.matmul(A, Pk_square) + tf.matmul(B, Pk) + C

            if self.criterion_choice_index == 4:
                # Combination criterion
                stop = self.stopping_criterion(Xk=X, Xk_prev=Xk_prev,
                                               Xk_prev_prev=Xk_prev_prev, Rk=Rk)
            elif self.criterion_choice_index == 3:
                stop = self.stopping_criterion(Xk=None, Xk_prev=None, Rk=Rk)
            elif self.criterion_choice_index == 2:
                stop = self.stopping_criterion(Xk=X, Xk_prev=Xk_prev, Xk_prev_prev=Xk_prev_prev)
            else:
                stop = self.stopping_criterion(Xk=X, Xk_prev=Xk_prev)

            # Check if stopping criterion is met for all batch elements
            if tf.reduce_all(stop):
                break

            # Store previous iterates
            Xk_prev_prev = Xk_prev
            Xk_prev = X

            # Compute (I - Y X)
            I_minus_YX = I - tf.matmul(Y, X)
            I_minus_XY = I - tf.matmul(X, Y)

            # Compute inverses
            try:
                inv_I_minus_YX = tf.linalg.inv(I_minus_YX)
                inv_I_minus_XY = tf.linalg.inv(I_minus_XY)
            except tf.errors.InvalidArgumentError:
                raise ValueError("Matrix inversion failed in SDA1 during iteration: "
                                 "I - Y*X or I - X*Y is singular. "
                                 "Please provide a different initial guess P0.")

            # Update E, F, X, Y
            E = tf.matmul(tf.matmul(E, inv_I_minus_YX), E)
            F = tf.matmul(tf.matmul(F, inv_I_minus_XY), F)
            X = X + tf.matmul(tf.matmul(F, inv_I_minus_XY), tf.matmul(X, E))
            Y = Y + tf.matmul(tf.matmul(E, inv_I_minus_YX), tf.matmul(Y, F))

            k += 1  # Increment iteration counter

        # Return X + P0
        Pk = X + P0
        return Pk

    def sda2(self):
        """
        Solve the quadratic matrix equation using SDA2 algorithm.

        Returns:
            Solution matrix P_k of shape (b, n, n)
        """
        # Algorithm SDA2 with inputs A, B, C.

        A = self.A
        B = self.B
        C = self.C

        batch_size = tf.shape(A)[0]
        n = tf.shape(A)[1]

        I = tf.eye(n, batch_shape=[batch_size], dtype=A.dtype)
        k = 0  # Iteration counter

        # Initialize variables
        X = tf.zeros_like(A)  # X0 = 0
        Y = -B
        E = -C
        F = -A

        Xk_prev = X
        Xk_prev_prev = None  # For Kahan's criterion
        Rk = None  # For residual norm criterion

        while True:
            # Compute the stopping criterion
            if self.criterion_choice_index in [3, 4]:
                # Compute residual norm: Rk = A Pk^2 + B Pk + C
                X_B = X + B
                try:
                    inv_X_B = tf.linalg.inv(X_B)
                except tf.errors.InvalidArgumentError:
                    raise ValueError("Matrix inversion failed in SDA2: X + B is singular. "
                                     "Please use SDA1 with an appropriate initial guess P0.")
                Pk = -tf.matmul(inv_X_B, C)
                Pk_square = tf.matmul(Pk, Pk)
                Rk = tf.matmul(A, Pk_square) + tf.matmul(B, Pk) + C

            if self.criterion_choice_index == 4:
                # Combination criterion
                stop = self.stopping_criterion(Xk=X, Xk_prev=Xk_prev,
                                               Xk_prev_prev=Xk_prev_prev, Rk=Rk)
            elif self.criterion_choice_index == 3:
                stop = self.stopping_criterion(Xk=None, Xk_prev=None, Rk=Rk)
            elif self.criterion_choice_index == 2:
                stop = self.stopping_criterion(Xk=X, Xk_prev=Xk_prev, Xk_prev_prev=Xk_prev_prev)
            else:
                stop = self.stopping_criterion(Xk=X, Xk_prev=Xk_prev)

            # Check if stopping criterion is met for all batch elements
            if tf.reduce_all(stop):
                break

            # Store previous iterates
            Xk_prev_prev = Xk_prev
            Xk_prev = X

            # Compute (X - Y)
            X_minus_Y = X - Y
            try:
                inv_X_minus_Y = tf.linalg.inv(X_minus_Y)
            except tf.errors.InvalidArgumentError:
                raise ValueError("Matrix inversion failed in SDA2 during iteration: "
                                 "X - Y is singular. Please use SDA1 with an appropriate initial guess P0.")

            # Update E, F, X, Y
            E = tf.matmul(tf.matmul(E, inv_X_minus_Y), E)
            F = tf.matmul(tf.matmul(F, inv_X_minus_Y), F)
            X = X - tf.matmul(tf.matmul(F, inv_X_minus_Y), E)
            Y = Y + tf.matmul(tf.matmul(E, inv_X_minus_Y), F)

            k += 1  # Increment iteration counter

        # Return Pk = - (X + B)^{-1} C
        X_B = X + B
        try:
            inv_X_B = tf.linalg.inv(X_B)
        except tf.errors.InvalidArgumentError:
            raise ValueError("Matrix inversion failed in SDA2 at final step: X + B is singular. "
                             "Please use SDA1 with an appropriate initial guess P0.")
        Pk = -tf.matmul(inv_X_B, C)

        return Pk
