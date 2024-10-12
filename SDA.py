    @staticmethod
    def solve_cubic(a, b, c, d):
        p = (3*a*c - b**2) / (3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
        delta = (q**2 / 4 + p**3 / 27)

        def compute_one_real_root():
            u = tf.pow(-q/2 + tf.sqrt(delta), 1/3)
            v = tf.pow(-q/2 - tf.sqrt(delta), 1/3)
            x1 = u + v - b / (3*a)
            return tf.stack([x1])

        def compute_three_real_roots():
            phi = tf.acos(-q / (2 * tf.sqrt(-(p**3) / 27)))
            root1 = 2 * tf.sqrt(-p / 3) * tf.cos(phi / 3) - b / (3*a)
            root2 = 2 * tf.sqrt(-p / 3) * tf.cos((phi + 2 * 3.141592653589793) / 3) - b / (3*a)
            root3 = 2 * tf.sqrt(-p / 3) * tf.cos((phi + 4 * 3.141592653589793) / 3) - b / (3*a)
            return tf.stack([root1, root2, root3])

        def compute_double_root():
            u = tf.pow(-q/2, 1/3)
            x1 = 2*u - b / (3*a)
            x2 = -u - b / (3*a)
            return tf.stack([x1, x2])

        roots = tf.cond(delta > 0, compute_one_real_root, 
                        lambda: tf.cond(delta < 0, compute_three_real_roots, compute_double_root))
        return roots

    def optimize(self):
        def process_batch(batch):
            A, B, C = batch
            n = tf.shape(A)[0]

            def compute_pj(j):
                a_j = A[:, j]
                b_j = B[:, j]
                c_j = C[:, j]

                a_bar = tf.reduce_sum(tf.square(a_j))
                b_bar = 2 * tf.reduce_sum(a_j * b_j)
                c_bar = tf.reduce_sum(tf.square(b_j)) + 2 * tf.reduce_sum(a_j * c_j)
                d_bar = 2 * tf.reduce_sum(b_j * c_j)
                e_bar = tf.reduce_sum(tf.square(c_j))

                def objective(p_j):
                    return a_bar * p_j**4 + b_bar * p_j**3 + c_bar * p_j**2 + d_bar * p_j + e_bar

                roots = self.solve_cubic(4*a_bar, 3*b_bar, 2*c_bar, d_bar)

                candidates = tf.concat([roots, [-self.rho, self.rho]], axis=0)
                values = tf.map_fn(objective, candidates)
                min_index = tf.argmin(values)
                return candidates[min_index]

            p_j_star = tf.map_fn(compute_pj, tf.range(n), dtype=tf.float32)
            return tf.linalg.diag(p_j_star)

        P_0 = tf.map_fn(process_batch, (self.A, self.B, self.C), dtype=tf.float32)
        return P_0
        
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
            solution = tf.map_fn(
                self.sda1_per_batch,
                elems=(A, B, C, P0)
            )
        # Implement SDA2
        elif self.algorithm_choice_index == 2:
            solution = tf.map_fn(
                self.sda2_per_batch,
                elems=(A, B, C)
            )
        else:
            raise ValueError("Invalid algorithm choice index.")
        return solution


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
        return Pk

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






More specifically: Substructure "type=tuple str=(<tf.Tensor: shape=(1, 5, 5), dtype=float64, numpy=
array([[[ 0.4459007 ,  1.43996715, -1.38644356,  0.51182528,
          0.87963893],
        [ 1.32904174,  2.07977811, -0.07583526,  0.60519562,
          0.95647552],
        [-0.65124187, -1.41262404, -3.02434557,  0.4175981 ,
          2.30271636],
        [-0.44071443,  0.20973944, -0.11783237,  1.14584635,
         -0.91954673],
        [ 0.7946478 , -0.03666153,  0.00494669,  0.80259986,
         -0.87223537]]])>, <tf.Tensor: shape=(1, 5, 5), dtype=float64, numpy=
array([[[ 2.22804923, -0.25682397,  0.08359474, -1.15861992,
         -0.02307564],
        [-0.92295232,  1.36665743,  0.09324357,  2.46719502,
         -0.12574691],
        [-1.00382139,  1.04095183,  0.93539273, -0.74975142,
         -0.42351937],
        [ 0.4950693 ,  0.64746207, -1.53577898,  0.28922091,
         -0.44998647],
        [-0.52597309,  2.25596801, -0.66511859, -0.18578899,
          1.2145866 ]]])>, <tf.Tensor: shape=(1, 5, 5), dtype=float64, numpy=
array([[[ 0.64368884,  0.70454497,  0.44394513, -0.82039107,
          0.14167135],
        [ 0.3373044 ,  0.66902667,  0.16668209,  0.46974965,
         -1.62653995],
        [ 0.5459081 , -0.59633392, -0.47746499, -0.91276906,
          1.73179862],
        [-0.03041607,  0.71864677,  0.21071723,  0.96983939,
          0.91086043],
        [ 1.25750075,  1.9858666 , -0.59410331,  0.38335737,
          0.42719292]]])>, <tf.Tensor: shape=(1, 5, 5), dtype=float64, numpy=
array([[[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]]])>)" is a sequence, while substructure "type=EagerTensor str=tf.Tensor(
[[-0.07253977 -0.35932881 -0.25634917 -0.00391021  0.44534032]
 [ 0.05738737  0.0724994   0.27577912 -0.09401044 -0.28451844]
 [ 0.170803    0.5400588   0.22769675  0.50460562  0.31510738]
 [ 0.09546884 -0.15554407 -0.25575423 -0.31576579  0.75390171]
 [-0.72466069 -1.02313565 -0.01886166  0.14551651  0.65513609]], shape=(5, 5), dtype=float64)" is not
Entire first structure:
(., ., ., .)
Entire second structure:
