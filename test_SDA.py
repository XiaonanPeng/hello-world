from SDA import SDA 
import tensorflow as tf 

class TestSDA:
    def __init__(self, sda_solver):
        self.sda_solver = sda_solver

    def test_solve_cubic(self, a, b, c, d):
        # Test solve_cubic method with given coefficients
        roots = self.sda_solver.solve_cubic(a, b, c, d)
        
        # Example expected roots for demonstration
        expected_roots = [1.0, 2.0, 3.0]
        
        # Check if computed roots match expected roots
        for root in expected_roots:
            assert any(tf.abs(roots - root) < 1e-6), f"Root {root} not found"

    def test_solve(self, A, B, C, P0=None):
        # Test solve method with given matrices
        solution, iterations = self.sda_solver.solve(A, B, C, P0)
        
        # Check if the solution satisfies the quadratic equation
        residual = tf.matmul(A, tf.matmul(solution, solution)) + tf.matmul(B, solution) + C
        residual_norm = tf.norm(residual, axis=[-2,-1])

        print('Solution P:', solution.numpy())
        print('Iterations:', iterations.numpy())
        print('Residual Norm:', residual_norm.numpy())

    def run_all_tests(self, a, b, c, d, A, B, C, P0=None):
        # Run all tests with provided inputs
        try:
            self.test_solve_cubic(a, b, c, d)
            print("test_solve_cubic passed")
        except AssertionError as e:
            print(f"test_solve_cubic failed: {e}")     
        
        self.test_solve(A, B, C, P0)


sda_solver = SDA(tol = 1e-12,
    algorithm_choice_index = 1,
    criterion_choice_index = 2,
    rho = 0.9999)

# Define matrices and initial guess
A = tf.constant([[[0.0, 0.25], [0.0, 0.25]]], dtype=tf.float32)
B = tf.constant([[[-0.75, 0.25], [0.25, -0.75]]], dtype=tf.float32)
C = tf.constant([[[0.25, 0.0], [0.25, 0.0]]], dtype=tf.float32)
P0 = tf.constant([[[0.0, 0.0], [0.0, 0.0]]], dtype=tf.float32)

# Test values for solve_cubic and stopping_criterion
a, b, c, d = tf.constant(1.0), tf.constant(-6.0), tf.constant(11.0), tf.constant(-6.0)

tester = TestSDA(sda_solver)
tester.run_all_tests(a, b, c, d, A, B, C)

dtype = tf.float64
tf.random.set_seed(2)
# Define dimensions and batch size
batch_size = 1
n = 5

A1 = tf.random.normal((batch_size, n, n), dtype=dtype)
B1 = tf.random.normal((batch_size, n, n), dtype=dtype)
C1 = tf.random.normal((batch_size, n, n), dtype=dtype)
rho = 0.9
P01 = tf.eye(n,batch_shape=[batch_size], dtype=dtype) * rho
#tester.test_solve(A1, B1, C1, P01)
