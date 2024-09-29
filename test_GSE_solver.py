import tensorflow as tf
from GSE_solver import GSE

def test_GSE_solver():
    """
    Test the generalized Sylvster solver by comparing its output with a known solution.
                        E + AX + CXD = 0
    """
    # set random seed
    tf.random.set_seed(1)
    # set matrix size and dtype
    n = 20
    b = 5
    dtype = tf.float32
    # generate randomly the coefficient matrices
    A = tf.random.normal((b, n, n), dtype=dtype)
    C = tf.random.normal((b, n, n), dtype=dtype)
    D = tf.random.normal((b, n, n), dtype=dtype)
    X_true = tf.random.normal((b, n, n), dtype=dtype)

    # compute the constant matrix E
    E = -tf.matmul(A, X_true) - tf.matmul(tf.matmul(C, X_true), D)

    # compute the X from GSE_solver
    solver = GSE()
    X_computed = solver.GSE_solver1(A, C, D, E)

    # compute error
    diff = X_computed - X_true
    diff_norm = tf.norm(diff, axis=[1,2])
    true_norm = tf.norm(X_true, axis=[1,2])
    relative_error = diff_norm / true_norm
    print(f"Relative Error:{relative_error.numpy()}")

if __name__ == "__main__":
    test_GSE_solver()
