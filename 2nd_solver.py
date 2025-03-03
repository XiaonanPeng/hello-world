import tensorflow as tf

# ==============================================================================
# SecondOrderDSGESolver
# ==============================================================================

class SecondOrderDSGESolver:
    """
    Solver class for computing the second-order perturbation of a DSGE model.
    
    This class constructs and solves two batched linear systems:
    
    (1) The state perturbation system. The second-order residual F_xx, evaluated at the 
    steady state, can be written as:
    
      F_xx(bar_x, 0) = 
         T1^i_{jk} + [H_y]^i_α [g_{xx}]^α_{βδ} [h_x]^δ_k [h_x]^β_j +
         [H_{y'}]^i_α [g_x]^α_β [h_{xx}]^β_{jk} +
         T4^i_{jk} + [H_x]^i_β [h_{xx}]^β_{jk} = 0.
    
    In the above, the terms T1 and T4 represent parts that are known (depend only on the 
    first-order solution and higher-order derivatives that do not multiply the unknowns).
    The unknowns are vec(g_{xx}) and vec(h_{xx}). This function assembles a batched linear 
    system A_total * u = -b_total (with u = [vec(gxx); vec(hxx)]) using high‐level TensorFlow 
    operations (tf.einsum, tf.reshape, tf.concat) to avoid explicit loops.
    
    (2) The shock (sigma) perturbation system. It is given by:
    
         ([H_{y'} + H_y,   H_{y'}g_x + H_{x'}]) * [g_{σσ}; h_{σσ}] + B = 0.
    
    Here, the vector B is defined as:
    
         B^i = sum( ([ [g_x; I]^T * ( [H_{y'y'}, H_{y'x'}; H_{x'y'}, H_{x'x'}]^i * [g_x; I] 
                          + [H_y]^i * g_{xx}) ] ⊙ (η Σ η^T) ) )
    
    where the unknown vector is u_sigma = [g_{σσ}; h_{σσ}], with total dimension n (n = n_y + n_x).
    
    The required keys in the `derivatives` dictionary are (all with batch dimension B):
      - "H_y":             [B, n, n_y]
      - "H_yprime":        [B, n, n_y]
      - "H_yprime_yprime": [B, n, n_y, n_y]
      - "H_yprime_y":      [B, n, n_y, n_y]
      - "H_yprime_xprime": [B, n, n_y, n_x]
      - "H_yprime_x":      [B, n, n_y, n_x]
      - "H_yy":            [B, n, n_y, n_y]
      - "H_yx":            [B, n, n_y, n_x]
      - "H_xprime_y":      [B, n, n_x, n_y]
      - "H_xprime_xprime": [B, n, n_x, n_x]
      - "H_xprime_x":      [B, n, n_x, n_x]
      - "H_x":             [B, n, n_x]
    
    In addition, for the sigma system the following are required:
      - "eta":             [B, n_x, n_e]
      - "Sigma":           [B, n_e, n_e]
    
    The first_order solution (passed via first_order_sol) must include:
      - "g_x":             [B, n_y, n_x]
      - "h_x":             [B, n_x, n_x]
    """
    def __init__(self, derivatives: dict, first_order_sol: dict):
        self.derivs = derivatives
        self.first_order = first_order_sol

    def solve(self):
        """
        Solve for the second-order coefficients.
        
        Returns:
            hxx: Tensor of shape [B, n_x, n_x, n_x] – second order state derivative tensor.
            gxx: Tensor of shape [B, n_y, n_x, n_x] – second order policy derivative tensor.
            hss: Tensor of shape [B, n_x] – sigma derivative for state variables.
            gss: Tensor of shape [B, n_y] – sigma derivative for policy variables.
        """
        hxx, gxx = self._solve_hxx_gxx()
        hss, gss = self._solve_hss_gss()
        return hxx, gxx, hss, gss

    def _solve_hxx_gxx(self):
        """
        Constructs and solves the batched linear system for the second-order derivatives w.r.t. x.
        
        The unknowns appear in terms:
          (a) g_{xx} part (multiplied by [H_y])
          (b) h_{xx} part (multiplied by [H_{y'}] and [H_x])
          
        The system is constructed as:
          
            A_total * u = -b_total,
          
        where u = [vec(gxx); vec(hxx)]. All operations use tf.einsum, tf.concat, and tf.reshape.
        """
        # Unpack tensors from first-order solution and derivatives.
        H_y      = self.derivs["H_y"]         # [B, n, n_y]
        H_yprime = self.derivs["H_yprime"]      # [B, n, n_y]
        H_x      = self.derivs["H_x"]           # [B, n, n_x]
        g_x      = self.first_order["g_x"]      # [B, n_y, n_x]
        h_x      = self.first_order["h_x"]      # [B, n_x, n_x]
  
        batch_size = tf.shape(g_x)[0]
        n_y = tf.shape(g_x)[1]
        n_x = tf.shape(g_x)[2]
        n = n_y + n_x  # Total number of equations
  
        # -------------------------------
        # Construct the known term b_total.
        # Note: The following is an illustration. In practice, all terms from F_xx that do not multiply the unknowns
        # must be assembled here. For example, terms involving derivatives like H_yprime_yprime, H_yprime_y, etc.
        t1 = tf.einsum('bian,bng,bgk->binjk', self.derivs["H_yprime_yprime"], g_x, h_x)
        t2 = tf.einsum('bian,bng->binjk', self.derivs["H_yprime_y"], g_x)
        t3 = tf.einsum('biax,bxk->binjk', self.derivs["H_yprime_xprime"], h_x)
        t4 = self.derivs["H_yprime_x"]  # assumed shape [B, n, n_y, n_x]
  
        known_part = t1 + t2 + t3 + t4
        b_total = tf.einsum('binjk,bga,bb->binjk', known_part, g_x, h_x)
        b_total_flat = tf.reshape(b_total, [batch_size, n * n_x * n_x])
  
        # -------------------------------
        # Build the coefficient matrix A_total.
        # Block for g_{xx}:
        # A_g^{i,j,k; α,β,δ} = [H_y]^i_α * ( [h_x]^β_j [h_x]^δ_k + δ_{β,j} δ_{δ,k} )
        A_g1 = tf.einsum('bin,bpj,bqk->binpjqk', H_y, h_x, h_x)
        delta = tf.eye(n_x, dtype=h_x.dtype)
        A_g2 = tf.einsum('bin,jk->binjk', H_y, delta)
        A_g_total = A_g1 + A_g2  # shape: [B, n, n_y, n_x, n_x]
        A_g_flat = tf.reshape(A_g_total, [batch_size, n * n_x * n_x, n_y * n_x * n_x])
  
        # Block for h_{xx}:
        # A_h^{i,j,k; β,j',k'} = ( [H_yprime]^i_α [g_x]^α_β + [H_x]^i_β ) * δ_{j,j'} δ_{k,k'}
        tmp = tf.einsum('bin,bna->bia', H_yprime, g_x)
        A_h_coeff = tmp + H_x  # shape: [B, n, n_x]
        delta_h = tf.eye(n_x, dtype=h_x.dtype)
        delta_h = tf.reshape(delta_h, [1, 1, n_x, n_x])
        A_h_total = A_h_coeff[..., None, None] * delta_h  # shape: [B, n, n_x, n_x, n_x]
        A_h_flat = tf.reshape(A_h_total, [batch_size, n * n_x * n_x, n_x * n_x * n_x])
  
        # Concatenate both blocks along the variable dimension:
        A_total = tf.concat([A_g_flat, A_h_flat], axis=-1)  # shape: [B, n*n_x*n_x, (n_y*n_x*n_x + n_x*n_x*n_x)]
  
        # Solve the linear system:
        u = tf.linalg.solve(A_total, -tf.expand_dims(b_total_flat, -1))  # [B, n*n_x*n_x, 1]
  
        n_g = tf.shape(A_g_flat)[-1]  # = n_y*n_x*n_x
        u_g = u[:, :, :n_g]
        u_h = u[:, :, n_g:]
  
        # Reshape the solution components.
        gxx = tf.reshape(u_g, [batch_size, n_y, n_x, n_x])
        hxx = tf.reshape(u_h, [batch_size, n_x, n_x, n_x])
        return hxx, gxx

    def _solve_hss_gss(self):
        """
        Constructs and solves the batched linear system for the second-order derivatives with respect to σ.
        
        The system is given by:
        
          ([H_{y'} + H_y,  H_{y'}g_x + H_{x'}]) * [g_{σσ}; h_{σσ}] + B = 0,
        
        where the vector B (with entries B^i) is computed as
        
          B^i = sum( ([g_x; I]^T * ( [H_{y'y'}, H_{y'x'}; H_{x'y'}, H_{x'x'}]^i * [g_x; I] + [H_y]^i * g_{xx})
                ⊙ (η Σ η^T) )
        
        This method builds the coefficient matrix A_sigma and known vector B, solves for u_sigma,
        then splits u_sigma into g_{σσ} and h_{σσ}.
        """
        coeff_mat_y_next = self.first_order["coeff_mat_y_next"]  # [B, n, n_y]
        coeff_mat_y      = self.first_order["coeff_mat_y"]       # [B, n, n_y]
        coeff_mat_x_next = self.first_order["coeff_mat_x_next"]   # [B, n, n_x]
        g_x = self.first_order["g_x"]                              # [B, n_y, n_x]
        H_y = self.derivs["H_y"]                                   # [B, n, n_y]
  
        batch_size = tf.shape(g_x)[0]
        n_y = tf.shape(g_x)[1]
        n_x = tf.shape(g_x)[2]
        n = n_y + n_x
  
        # Build the σ system coefficient matrix.
        A_left = coeff_mat_y_next + coeff_mat_y      # [B, n, n_y]
        A_right = tf.einsum('bin,bnj->bij', coeff_mat_y_next, g_x) + coeff_mat_x_next  # [B, n, n_x]
        A_sigma = tf.concat([A_left, A_right], axis=-1)  # [B, n, n_y+n_x] which is [B, n, n]
  
        # Build the right-hand side vector B_sigma.
        I_state = tf.eye(n_x, batch_shape=[batch_size], dtype=g_x.dtype)  # [B, n_x, n_x]
        g_x_stack = tf.concat([g_x, I_state], axis=1)  # [B, n_y+n_x, n_x]
  
        M_top = tf.concat([self.derivs["H_yprime_yprime"], self.derivs["H_yprime_xprime"]], axis=-1)
        M_bot = tf.concat([self.derivs["H_xprime_y"], self.derivs["H_xprime_xprime"]], axis=-1)
        M = tf.concat([M_top, M_bot], axis=1)  # assumed shape: [B, n, n, n]
  
        tmp = tf.einsum('bijk,bkj->bij', M, g_x_stack)  # [B, n, n_x]
        E = tf.einsum('bji,bij->bi', g_x_stack, tmp)      # [B, n]
  
        # In practice, F = [H_y]^i * g_{xx} is computed using the solved gxx.
        # Here, we use a placeholder (zeros) since the actual gxx must be inserted.
        dummy_gxx = tf.zeros([batch_size, n_y, n_x, n_x], dtype=g_x.dtype)
        F = tf.einsum('bin,bnjk->bi', H_y, dummy_gxx)  # [B, n]
  
        B_sigma = E + F  # [B, n]
  
        # Multiply by an eta-based term.
        eta = self.derivs["eta"]         # [B, n_x, n_e]
        Sigma = self.derivs["Sigma"]     # [B, n_e, n_e]
        etaSigma = tf.einsum('bij,bjk->bik', eta, Sigma)
        etaSigmaEta = tf.einsum('bij,bkj->bik', etaSigma, eta)
        etaTerm = tf.reduce_sum(etaSigmaEta, axis=[-2, -1])   # [B]
        B_sigma = B_sigma * tf.reshape(etaTerm, [batch_size, 1])
  
        # Solve for u_sigma.
        u_sigma = tf.linalg.solve(A_sigma, -tf.expand_dims(B_sigma, -1))  # [B, n, 1]
        u_sigma = tf.squeeze(u_sigma, axis=-1)  # [B, n]
  
        gss = u_sigma[:, :n_y]
        hss = u_sigma[:, n_y:]
        return hss, gss

# ==============================================================================
# SecondOrderDSGE (inherits from FirstOrderDSGE)
# ==============================================================================

class SecondOrderDSGE(FirstOrderDSGE):
    """
    SecondOrderDSGE extends the first-order DSGE solution by computing the second-order perturbation.
    
    This class inherits from FirstOrderDSGE so that all first-order solution components
    (such as h_x, g_x, coeff_mat_x, coeff_mat_y, coeff_mat_x_next, coeff_mat_y_next and the associated 
    noise functions) are available without recomputation.
    
    In addition, it uses a derivative-parsing function (e.g., parse_derivative_fn) to extract all 
    necessary second-order derivative mappings. In particular, the functions must provide the following:
        H_{yprime_yprime}, H_{yprime_y}, H_{yprime_xprime}, H_{yprime_x},
        H_{yy}, H_{yx}, H_{xprime_y}, H_{xprime_xprime}, H_{xprime_x}, and H_{xx}
    as needed in the second-order system for F_xx, plus the eta and Sigma for the sigma system.
    
    Finally, the method solve_second_order(paras_tensor) uses the inherited first-order solve() 
    and the evaluated second-order derivatives to call SecondOrderDSGESolver and return hxx, gxx, 
    hss, and gss.
    """
    def __init__(self, paras, state_vars, policy_vars, shock_vars, equilibrium_equations,
                 exogenous_processes, observation_eqs=None, configs=None):
        # Inherit initialization from FirstOrderDSGE.
        super().__init__(paras, state_vars, policy_vars, shock_vars,
                         equilibrium_equations, exogenous_processes, observation_eqs, configs)
        self.dtype = tf.float32

        # The first-order derivative functions and solutions (e.g. coeff_mat_y_next, etc.) are inherited.
        # Now extract the second-order derivative mappings.
        self.second_order_deriv_fns = self.extract_second_order_equation_matrix_fns()

    def extract_second_order_equation_matrix_fns(self):
        """
        Extract the second-order derivative functions for the DSGE model with respect to 
        (x, y, x', y').
        
        This function should leverage your derivative-parsing function (e.g., parse_derivative_fn)
        and return a dictionary containing all relevant second-order derivative mappings:
        
            "H_yprime_yprime", "H_yprime_y", "H_yprime_xprime", "H_yprime_x",
            "H_yy", "H_yx", "H_xprime_y", "H_xprime_xprime", "H_xprime_x", "H_x_x",
            "eta", and "Sigma".
        
        The expansion point is given by the steady state values (state_steady and policy_steady).
        """
        all_args = (self.state_vars, self.state_steady,
                    self.policy_vars, self.policy_steady, self.paras)
        expansion_point = (self.state_steady, self.state_steady,
                           self.policy_steady, self.policy_steady, None)
  
        deriv_fns = dict()
        deriv_fns["H_yprime_yprime"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.policy_next, self.policy_next],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_yprime_y"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.policy_next, self.policy_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_yprime_xprime"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.policy_next, self.state_next],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_yprime_x"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.policy_next, self.state_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_yy"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.policy_vars, self.policy_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_yx"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.policy_vars, self.state_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_xprime_y"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.state_next, self.policy_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_xprime_xprime"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.state_next, self.state_next],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_xprime_x"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.state_next, self.state_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        deriv_fns["H_x_x"] = parse_derivative_fn(
            all_args, arg_sequence_to_be_diff=[self.state_vars, self.state_vars],
            exprs=(self.equilibrium_residuals,), expansion_point=expansion_point,
            num_batch_dim=1, dtype=self.dtype)
        # For the sigma system, include eta and Sigma.
        deriv_fns["eta"] = self.noise_mat_exogenous(self.paras)  # should yield [B, n_x, n_e]
        deriv_fns["Sigma"] = get_sigma_tensor()  # Replace with your actual sigma tensor function
  
        return deriv_fns

    def solve_second_order(self, paras_tensor: tf.Tensor):
        """
        Given a parameter tensor paras_tensor of shape [B, dim_param], perform the following:
        
          1. Use the inherited solve() from FirstOrderDSGE to obtain the first-order solution: h_x and g_x.
          2. Use the parsed second-order derivative functions to evaluate all second-order derivatives.
          3. Call SecondOrderDSGESolver with the evaluated derivatives and the first-order solution,
             and obtain the second-order coefficients.
        
        Returns:
            hxx, gxx, hss, gss as defined in SecondOrderDSGESolver.
        """
        # Obtain first order solution using inherited solve() method.
        (h_x, g_x), _ = self.solve(paras_tensor)
        first_order_sol = {
            "g_x": g_x,    # shape: [B, n_y, n_x]
            "h_x": h_x,    # shape: [B, n_x, n_x]
            "coeff_mat_y_next": self.coeff_mat_y_next(paras_tensor),  # H_{y'}
            "coeff_mat_y": self.coeff_mat_y(paras_tensor),            # H_y
            "coeff_mat_x_next": self.coeff_mat_x_next(paras_tensor),  # H_{x'}
        }
  
        # Evaluate second-order derivative functions.
        deriv_evaluated = {}
        for key, fn in self.second_order_deriv_fns.items():
            if callable(fn):
                deriv_evaluated[key] = fn(paras_tensor)
            else:
                deriv_evaluated[key] = fn
  
        # Solve for second order coefficients.
        solver = SecondOrderDSGESolver(deriv_evaluated, first_order_sol)
        hxx, gxx, hss, gss = solver.solve()
        return hxx, gxx, hss, gss


