import tensorflow as tf
class FirstOrderDSGE(BaseDSGE):
    """A first-order subclass of DSGE model.
    Specifically, this includes the implementation of the following methods:
        - extract_1st_order_matrix_fns(): construct the function mapping from parameter to matrices;
        - obtain_QME_matrices(): mapping a parameter tensor to matrices in 1st order equations;
        - solve(): solve the linear REE to obtain policy and transition matrices.
    """

    def __init__(
        self,
        paras: Union[Tuple, List],
        state_vars: Union[Tuple, List],
        policy_vars: Union[Tuple, List],
        shock_vars: Union[Tuple, List],
        equilibrium_equations: Union[Tuple, List],
        exogenous_processes: Union[Tuple, List],
        observation_eqs: Union[Tuple, List] = None,
        configs: NamedTuple = None,
    ):
        super().__init__(
            paras,
            state_vars,
            policy_vars,
            shock_vars,
            equilibrium_equations,
            exogenous_processes,
            observation_eqs,
            configs,
        )

        (
            self.coeff_mat_x,
            self.coeff_mat_y,
            self.coeff_mat_x_next,
            self.coeff_mat_y_next,
        ) = self.extract_1st_order_equation_matrix_fns()
        self.noise_mat_exogenous = self.extrac_1st_order_noise_matrix_fn()
        self.residual_fn_perturbation = self.parsing_1st_order_residual_fns()

        self.solver = SDA(solve_config=self.configs.solve)
        self.grad_solver = GradientSolverBase((tf.float32, tf.complex64))

    def parsing_1st_order_residual_fns(self, num_batch_dim: int = 1):
        residual_exprs = list()
        args = self.state_vars + self.state_next + self.policy_vars + self.policy_next
        for exprs in get_derivative(args, self.equilibrium_residuals):  # [num_of_eqs, num_of_args]
            append_args = list()
            for arg, expr in zip(args, exprs):  # [num_of_args]
                constant_expr = expr
                if isinstance(expr, sp.Expr):
                    for v in args:
                        constant_expr = replace_expression(constant_expr, to_replace=v, replace_with=0)
                append_args.append(constant_expr * arg)
            residual_exprs.append(sp.simplify(Add(*append_args)))  # [num_of_eqs]

        exprs = (residual_exprs,)
        args = (self.paras, self.state_vars, self.state_next, self.policy_vars, self.policy_next)
        return parse_batchable_fn(args, exprs, num_batch_dim=num_batch_dim)

    def extract_1st_order_equation_matrix_fns(self):
        """Obtain all first order derivative matrices."""
        exprs = (self.equilibrium_residuals + [eq.rhs - eq.lhs for eq in self.exogenous_expectations],)
        all_args = (
            self.state_vars,
            self.state_next,
            self.policy_vars,
            self.policy_next,
            self.paras,
        )
        expansion_point = (
            self.state_steady,
            self.state_steady,
            self.policy_steady,
            self.policy_steady,
            None,
        )

        coeff_mat_x = parse_derivative_fn(
            all_args,
            arg_sequence_to_be_diff=[self.state_vars],
            exprs=exprs,
            expansion_point=expansion_point,
            num_batch_dim=1,
            dtype=self.dtype,
        )
        coeff_mat_y = parse_derivative_fn(
            all_args,
            arg_sequence_to_be_diff=[self.policy_vars],
            exprs=exprs,
            expansion_point=expansion_point,
            num_batch_dim=1,
            dtype=self.dtype,
        )
        coeff_mat_x_next = parse_derivative_fn(
            all_args,
            arg_sequence_to_be_diff=[self.state_next],
            exprs=exprs,
            expansion_point=expansion_point,
            num_batch_dim=1,
            dtype=self.dtype,
        )
        coeff_mat_y_next = parse_derivative_fn(
            all_args,
            arg_sequence_to_be_diff=[self.policy_next],
            exprs=exprs,
            expansion_point=expansion_point,
            num_batch_dim=1,
            dtype=self.dtype,
        )

        return (
            coeff_mat_x,
            coeff_mat_y,
            coeff_mat_x_next,
            coeff_mat_y_next,
        )

    def extrac_1st_order_noise_matrix_fn(self):
        """obtain a function for noise coefficient matrix on exogenous variables,
        which takes as input a parameter tensor, and outputs a square matrix of size [dim_exogenous, dim_exogenous].
        """
        exprs = (self.exogenous_noises,)  # 1-tuple with a length-dim_exogenous list
        all_args = (
            self.shock_vars,
            self.paras,
        )
        expansion_point = (
            tf.convert_to_tensor([0.0] * len(self.shock_vars), dtype=self.dtype),
            None,
        )

        return parse_derivative_fn(
            all_args,
            arg_sequence_to_be_diff=[self.shock_vars],
            exprs=exprs,
            expansion_point=expansion_point,
            num_batch_dim=1,
            dtype=self.dtype,
        )

    def obtain_QME_matrices(self, paras_tensor: tf.Tensor) -> Tuple[tf.Tensor]:
        """Construct a quadratic equation:
            AP^2 + BP + C = 0
        with:
            A = [  0,      H_{y'}]
            B = [H_{x'},   H_{y} ]
            C = [H_{x},      0   ]
        where H is the residual function of equilibrium and:
            P = [
                [h_{x}, 0],
                [g_{x}, 0],
            ]
        with g and h being the policy and state transition function, respectively.

        :param paras_tensor: shape = [B, dim_param], tensor containing values of parameters.
        """
        H_x = self.coeff_mat_x(paras_tensor)
        H_y = self.coeff_mat_y(paras_tensor)
        H_x_next = self.coeff_mat_x_next(paras_tensor)
        H_y_next = self.coeff_mat_y_next(paras_tensor)

        A = tf.concat([tf.zeros_like(H_x), H_y_next], axis=-1)
        B = tf.concat([H_x_next, H_y], axis=-1)
        C = tf.concat([H_x, tf.zeros_like(H_y)], axis=-1)
        return (A, B, C)

    def obtain_noise_matrix(self, paras_tensor: tf.Tensor) -> tf.Tensor:
        """obtain a batch of noise matrix"""
        noise_mat_exogenous = self.noise_mat_exogenous(paras_tensor)  # [B, dim_exo, dim_exo]
        noise_mat_endogenous = tf.repeat(tf.zeros_like(noise_mat_exogenous[..., 0:1, :]), self.dim_endogenous, axis=-2)
        noise_mat = tf.concat([noise_mat_endogenous, noise_mat_exogenous], axis=-2)  # [B, dim_state, dim_exogenous]
        return noise_mat

    def equation_mat_jacobian_to_parameters(self, paras_tensor: tf.Tensor) -> Tuple[tf.Tensor]:
        r"""compute Jacobian of matrices in quadratic equation with respect to parameters.
        The quadratic equation set is:
            AP^2 + BP + C = 0
        where coefficient matrices (A, B, C) are determined by parameters and derived steady state values.

        :param paras_tensor: shape = [B, p], parameters and steady states which determines equilibrium equations.
        :return tuple of jacobians.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(paras_tensor)
            A, B, C = self.obtain_QME_matrices(paras_tensor)
        dA_dTheta = tape.batch_jacobian(A, paras_tensor)  # [B, N, N, p]
        dB_dTheta = tape.batch_jacobian(B, paras_tensor)  # [B, N, N, p]
        dC_dTheta = tape.batch_jacobian(C, paras_tensor)  # [B, N, N, p]
        return (dA_dTheta, dB_dTheta, dC_dTheta)

    @tf.custom_gradient
    def solve(self, paras_tensor: tf.Tensor):
        A, B, C = self.obtain_QME_matrices(paras_tensor)

        P, indicator = self.solver.solve_P(A, B, C, None)  # [B, N, N]
        h_x, g_x = tf.split(
            P[..., : self.dim_state],
            num_or_size_splits=[self.dim_state, self.dim_policy],
            axis=-2,
            name="solution_matrix_split",
        )
        
class SecondOrderDSGESolver:
    def __init__(self,
                 # First-order derivatives (batch, n_eq, ...)
                 H_x, H_y,
                 H_xprime, H_yprime,
                 
                 # Second-order derivatives (batch, n_eq, ...)
                 H_xx, H_xy, H_yy,
                 H_xprimex, H_xprimey, H_xprimexprime,
                 H_yprimex, H_yprimey, H_yprimexprime,
                 
                 # First-order solution components
                 h_x,  # (batch, n_x, n_x)
                 g_x): # (batch, n_y, n_x)
        """Initializes second-order DSGE solver with model derivatives.
        
        Args:
            H_*: Model derivatives from equilibrium conditions
            h_x: First-order state transition matrix
            g_x: First-order decision rule matrix
        """
        # Store derivatives
        self.H_x, self.H_y = H_x, H_y
        self.H_xprime, self.H_yprime = H_xprime, H_yprime
        self.H_xx, self.H_xy, self.H_yy = H_xx, H_xy, H_yy
        self.H_xprimex, self.H_xprimey = H_xprimex, H_xprimey
        self.H_xprimexprime = H_xprimexprime
        self.H_yprimex, self.H_yprimey = H_yprimex, H_yprimey
        self.H_yprimexprime = H_yprimexprime
        
        # Store first-order solutions
        self.h_x = h_x
        self.g_x = g_x
        
        # Dimensions
        self.batch_size = tf.shape(H_x)[0]
        self.n_eq = H_x.shape[1]  # Number of equations
        self.n_x = h_x.shape[-1]  # Number of state variables
        self.n_y = g_x.shape[-2]  # Number of control variables
        
    def compute_second_order(self):
        """Main entry point: computes all second-order coefficients.
        
        Returns:
            Dictionary containing:
            - g_xx: (batch, n_y, n_x, n_x)
            - h_xx: (batch, n_x, n_x, n_x) 
            - g_σσ: (batch, n_y)
            - h_σσ: (batch, n_x)
        """
        g_xx, h_xx = self._solve_gxx_hxx()
        g_ss, h_ss = self._solve_gss_hss(g_xx, h_xx)
        
        return {
            'g_xx': g_xx,
            'h_xx': h_xx,
            'g_σσ': g_ss,
            'h_σσ': h_ss,
            # Cross terms are zero as shown in derivation
            'g_xσ': tf.zeros_like(g_xx[..., 0]),
            'h_xσ': tf.zeros_like(h_xx[..., 0])
        }

    def _solve_gxx_hxx(self):
        """Solves linear system for g_xx and h_xx coefficients."""
        # Construct coefficient matrix A (batch, n_eq*n_x², (n_y + n_x)*n_x²)
        A = self._build_gxx_coefficient_matrix()
        
        # Construct constant term C (batch, n_eq*n_x², 1)
        C = self._build_gxx_constant_term()
        
        # Solve linear system AX = -C
        X = tf.linalg.solve(A, -C)
        
        # Split and reshape solution
        g_xx = tf.reshape(X[..., :self.n_y*self.n_x**2, 0],
                         [self.batch_size, self.n_y, self.n_x, self.n_x])
        h_xx = tf.reshape(X[..., self.n_y*self.n_x**2:, 0],
                         [self.batch_size, self.n_x, self.n_x, self.n_x])
        return g_xx, h_xx

    def _build_gxx_coefficient_matrix(self):
        """Constructs coefficient matrix for g_xx/h_xx system."""
        # Term 1: H_y (I ⊗ h_xᵀ ⊗ h_xᵀ)
        term1 = tf.einsum('bij,bkl->bikjl', self.H_y, 
                         tf.eye(self.n_x, batch_shape=[self.batch_size]))
        term1 = tf.reshape(term1, [self.batch_size, self.n_eq*self.n_x**2, 
                                  self.n_y*self.n_x**2])
        
        # Term 2: [H_y' g_x + H_x'] (I ⊗ h_xᵀ)
        term2_base = tf.einsum('bijk,bkl->bijl', self.H_yprime, self.g_x) + self.H_xprime
        term2 = tf.einsum('bijk,bklm->bijlm', term2_base,
                         tf.eye(self.n_x, batch_shape=[self.batch_size]))
        term2 = tf.reshape(term2, [self.batch_size, self.n_eq*self.n_x**2,
                                  self.n_x*self.n_x**2])
        
        return tf.concat([term1, term2], axis=-1)

    def _build_gxx_constant_term(self):
        """Constructs constant term for g_xx/h_xx system."""
        # Term 1: H_y'y' g_x h_x ⊗ g_x h_x
        term1 = tf.einsum('bijk,bkl,bmn->bijnlm', self.H_yprimey, self.g_x, self.h_x)
        term1 = tf.einsum('bijnlm,bjpo->bipolm', term1, self.g_x @ self.h_x)
        
        # Term 2: H_y'x' g_x h_x ⊗ h_x
        term2 = tf.einsum('bijk,bkl,bmn->bijnlm', self.H_yprimex, self.g_x, self.h_x)
        term2 = tf.einsum('bijnlm,bjpo->bipolm', term2, self.h_x)
        
        # Term 3: H_yy g_x ⊗ g_x
        term3 = tf.einsum('bijk,bkl,bmn->bijlm', self.H_yy, self.g_x, self.g_x)
        
        # Term 4: H_x'x' h_x ⊗ h_x
        term4 = tf.einsum('bijk,bkl,bmn->bijlm', self.H_xprimexprime, self.h_x, self.h_x)
        
        # Combine terms and vectorize
        C = term1 + term2 + term3 + term4
        return tf.reshape(C, [self.batch_size, self.n_eq*self.n_x**2, 1])

    def _solve_gss_hss(self, g_xx, h_xx):
        """Solves linear system for g_σσ and h_σσ coefficients."""
        # Construct coefficient matrix (batch, n_eq, n_y + n_x)
        A = tf.concat([
            self.H_y + self.H_yprime,
            tf.einsum('bijk,bkl->bijl', self.H_yprime, self.g_x) + self.H_xprime
        ], axis=-1)
        
        # Construct constant term B (batch, n_eq, 1)
        B = self._build_gss_constant_term(g_xx, h_xx)
        
        # Solve linear system
        X = tf.linalg.solve(A, -B)
        
        # Split solution
        g_ss = X[..., :self.n_y, 0]
        h_ss = X[..., self.n_y:, 0]
        return g_ss, h_ss

    def _build_gss_constant_term(self, g_xx, h_xx):
        """Constructs constant term for σσ system."""
        # Term 1: H_y'y' g_x η Σ ηᵀ g_xᵀ
        term1 = tf.einsum('bijk,bkl,bmn,bno->bijo', 
                         self.H_yprimey, self.g_x, self.g_x, self.g_x)
        
        # Term 2: H_y'x' g_x η Σ ηᵀ
        term2 = tf.einsum('bijk,bkl,bmn->bijm', 
                         self.H_yprimex, self.g_x, self.g_x)
        
        # Term 3: H_y g_xx
        term3 = tf.einsum('bijk,bjklm->bilm', 
                         self.H_y, g_xx)
        
        # Term 4: H_x'x' η Σ ηᵀ
        term4 = tf.einsum('bijk,bkl,bmn->bijm', 
                         self.H_xprimexprime, self.g_x, self.g_x)
        
        # Sum all terms
        B = term1 + term2 + term3 + term4
        return tf.expand_dims(B, -1)
