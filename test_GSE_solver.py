import tensorflow as tf
from GSE_solver import GSE_solver

def test_GSE_solver():
    """
    Test the generalized Sylvster solver by comparing its output with a known solution.
                        E + AX + CXD = 0
    """
    # set random seed
    tf.random.set_seed(1)
    # set matrix size and dtype
    n = 5
    m = 5
    dtype = tf.float32
    # generate randomly the coefficient matrices
    A = tf.random.normal((n+m, n+m), dtype=dtype)
    C = tf.random.normal((n+m, n+m), dtype=dtype)
    D = tf.random.normal((n, n), dtype=dtype)
    X_true = tf.random.normal((n+m, n), dtype=dtype)

    # compute the constant matrix E
    E = -tf.matmul(A, X_true) - tf.matmul(tf.matmul(C,X_true), D)

    # compute the X from GSE_solver
    X_computed = GSE_solver(A, C, D, E)

    # compute error
    diff = X_computed - X_true
    diff_norm = tf.norm(diff)
    true_norm = tf.norm(X_true)
    relative_error = diff_norm / true_norm
    print(f"Relative Error:{relative_error.numpy():.2e}")

if __name__ == "__main__":
    test_GSE_solver()
