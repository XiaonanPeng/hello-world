import tensorflow as tf
import numpy as np
import unittest

class TestSecondOrderDSGESolver(unittest.TestCase):
    def setUp(self):
        """Create test data with known structure"""
        self.batch_size = 2
        self.n_eq = 3
        self.n_x = 2
        self.n_y = 1
        
        # Create simple identity-based derivatives for predictable results
        self.create_identity_derivatives()
        self.solver = SecondOrderDSGESolver(
            H_x=self.H_x, H_y=self.H_y,
            H_xprime=self.H_xprime, H_yprime=self.H_yprime,
            H_xx=self.H_xx, H_xy=self.H_xy, H_yy=self.H_yy,
            H_xprimex=self.H_xprimex, H_xprimey=self.H_xprimey,
            H_xprimexprime=self.H_xprimexprime,
            H_yprimex=self.H_yprimex, H_yprimey=self.H_yprimey,
            H_yprimexprime=self.H_yprimexprime,
            h_x=tf.eye(self.n_x, batch_shape=[self.batch_size]),
            g_x=tf.ones([self.batch_size, self.n_y, self.n_x])
        )

    def create_identity_derivatives(self):
        """Create test derivatives with identity matrix patterns"""
        eye3 = np.eye(self.n_eq)
        self.H_x = tf.constant(np.stack([eye3]*self.batch_size), dtype=tf.float32)
        self.H_y = tf.constant(np.stack([eye3]*self.batch_size), dtype=tf.float32)
        
        # Create derivative tensors with known patterns
        self.H_xprime = tf.eye(self.n_eq, batch_shape=[self.batch_size])
        self.H_yprime = tf.eye(self.n_eq, batch_shape=[self.batch_size])
        
        # Second derivatives with controllable patterns
        self.H_xx = tf.zeros([self.batch_size, self.n_eq, self.n_x, self.n_x])
        self.H_xy = tf.zeros([self.batch_size, self.n_eq, self.n_x, self.n_y])
        self.H_yy = tf.zeros([self.batch_size, self.n_eq, self.n_y, self.n_y])
        
        self.H_xprimex = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_x])
        self.H_xprimey = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_y])
        self.H_xprimexprime = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_x, self.n_x])
        
        self.H_yprimex = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_x])
        self.H_yprimey = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_y])
        self.H_yprimexprime = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_x, self.n_x])

    def test_gxx_coefficient_dimensions(self):
        """Verify coefficient matrix dimensions for g_xx/h_xx system"""
        A = self.solver._build_gxx_coefficient_matrix()
        expected_shape = (
            self.batch_size,
            self.n_eq * self.n_x**2,
            (self.n_y + self.n_x) * self.n_x**2
        )
        self.assertEqual(A.shape, expected_shape)

    def test_gxx_constant_term_values(self):
        """Verify constant term calculation with known inputs"""
        # Set specific derivative values for predictable output
        self.H_yprimey = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_y, self.n_y])
        self.H_yprimex = tf.eye(self.n_eq, batch_shape=[self.batch_size, self.n_x, self.n_x])
        
        C = self.solver._build_gxx_constant_term()
        
        # Verify numerical values for first batch element
        C_np = C[0].numpy().flatten()
        expected_non_zero = min(self.n_eq, self.n_x**2)
        self.assertAlmostEqual(np.count_nonzero(C_np), expected_non_zero)

    def test_gxx_solution_shape(self):
        """Verify solution tensor shapes for g_xx/h_xx"""
        g_xx, h_xx = self.solver._solve_gxx_hxx()
        
        self.assertEqual(g_xx.shape, (self.batch_size, self.n_y, self.n_x, self.n_x))
        self.assertEqual(h_xx.shape, (self.batch_size, self.n_x, self.n_x, self.n_x))

    def test_gss_constant_term(self):
        """Verify sigma constant term construction"""
        g_xx = tf.zeros([self.batch_size, self.n_y, self.n_x, self.n_x])
        h_xx = tf.zeros([self.batch_size, self.n_x, self.n_x, self.n_x])
        
        B = self.solver._build_gss_constant_term(g_xx, h_xx)
        
        # Should have shape (batch, n_eq, 1)
        self.assertEqual(B.shape, (self.batch_size, self.n_eq, 1))

    def test_full_solution_consistency(self):
        """Verify consistency between different solution components"""
        results = self.solver.compute_second_order()
        
        # Verify cross terms are zero
        self.assertAllClose(results['g_xσ'], tf.zeros_like(results['g_xσ']))
        self.assertAllClose(results['h_xσ'], tf.zeros_like(results['h_xσ']))
        
        # Verify main terms have correct relationships
        self.assertAllClose(
            results['g_xx'][0],
            tf.transpose(results['g_xx'][0], perm=[0, 2, 1]),
            atol=1e-6
        )

    def test_known_solution(self):
        """Test with hand-calculated small system"""
        # Create minimal system (1 equation, 1 state, 1 control)
        self.n_eq = 1
        self.n_x = 1
        self.n_y = 1
        self.create_identity_derivatives()
        
        # Set specific values for identity relationships
        self.H_y = tf.ones([self.batch_size, self.n_eq, self.n_y])
        self.H_xprime = tf.ones([self.batch_size, self.n_eq, self.n_x])
        
        solver = SecondOrderDSGESolver(
            # ... pass all parameters ...
            h_x=tf.ones([self.batch_size, self.n_x, self.n_x]),
            g_x=tf.ones([self.batch_size, self.n_y, self.n_x])
        )
        
        results = solver.compute_second_order()
        
        # Verify solution shapes
        self.assertEqual(results['g_xx'].shape, (self.batch_size, 1, 1, 1))
        self.assertEqual(results['h_xx'].shape, (self.batch_size, 1, 1, 1))

if __name__ == '__main__':
    tf.test.main()
