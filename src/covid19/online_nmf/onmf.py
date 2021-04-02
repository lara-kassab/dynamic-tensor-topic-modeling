# (setq python-shell-interpreter "./venv/bin/python")

import numpy as np
import scipy
from numpy import linalg as LA
from sklearn.decomposition import SparseCoder
from tqdm import tqdm

DEBUG = False


class Online_NMF:
    """ONMF for a generator of data matrices representing a time series."""

    def __init__(
        self,
        data_stack,
        num_topics=10,
        regularization_param=1,
        iterations=500,
        batch_size=100,
        beta=None,
        ini_dict=None,
    ):
        """Store arguments for Online_NMF instance.

        Args:
            X: time series of data matrices (iterable of matrices)
                Each matrix should have dimensions: data_dim (d) x samples (n).
            num_topics (int): number of columns in dictionary matrix W, where
                each column represents a topic/feature.
            regularization_param (scalar): the transform_alpha to be used in
                SparseCoder of sparse_code.
            iterations (int): number of iterations where each iteration is a
                call to step(...) for each time slice of the stack.
            batch_size (int): number random of columns of X that will be
                sampled during each iteration per slice.
        """
        self.data_stack = data_stack
        self.num_topics = num_topics
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_dict = ini_dict
        self.reg_param = regularization_param

        if beta is None:
            self.beta = 1
        else:
            self.beta = beta

        if self.reg_param < 0:
            raise Exception("The regularization_param must be at least 0.")

    def get_code_matrix(self, X, W=None):
        """Solve for code matrix H to such that W*H approximates X.

        Args:
            X (ndarry): data matrix with dimensions:
                data_dim (d) x samples (n)
            W (ndarry): dictionary matrix with dimensions:
                data_dim (d) x topics (r)

        Returns:
            H (ndarry): code matrix with dimensions:
                topics (r) x samples(n)
        """
        if W is None:
            raise Exception(
                """Missing required dictionary keyword argument W in sparse_code."""
            )

        # Solve least-squares problem for code matrix H.
        if self.reg_param == 0:
            return self.non_negative_ls(X, W)
        # Solve sparse coding problem.
        elif self.reg_param > 0:
            return self.sparse_code(X, W)

    def non_negative_ls(self, X, W):
        """Solve for code matrix H to such that W*H approximates X.

        Given data matrix X and dictionary matrix W, find a nonnegative code
        matrix H such that W*H approximates X.

        Args:
            X (ndarry): data matrix with dimensions:
                data_dim (d) x samples (n)
            W (ndarry): dictionary matrix with dimensions:
                data_dim (d) x topics (r)

        Returns:
            H (ndarry): code matrix with dimensions:
                topics (r) x samples(n)
        """
        H = np.zeros((W.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            H[:, i] = scipy.optimize.nnls(W, X[:, i].flatten())[0]
        return H

    def sparse_code(self, X, W):
        """Solve for code matrix H to such that W*H approximates X.

        Given data matrix X and dictionary matrix W, find a nonnegative code
        matrix H such that W*H approximates X with regularization to promote
        sparsity in H.

        Args:
            X (ndarry): data matrix with dimensions:
                data_dim (d) x samples (n)
            W (ndarry): dictionary matrix with dimensions:
                data_dim (d) x topics (r)

        Returns:
            H (ndarry): code matrix with dimensions:
                topics (r) x samples(n)
        """
        # TODO: look at more efficient implementation when
        # self.reg_param = regularization_param is 0 - should use numpy.linalg.lstsq
        # change name to get_topic distributions.
        if DEBUG:
            print("sparse_code")
            print("X.shape:", X.shape)
            print("W.shape:", W.shape, "\n")

        # Initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H.
        coder = SparseCoder(
            dictionary=W.T,
            transform_n_nonzero_coefs=None,
            transform_alpha=self.reg_param,
            transform_algorithm="lasso_lars",
            positive_code=True,
        )
        H = coder.transform(X.T)
        # Transpose H before returning to undo the preceding transpose on X.
        return H.T

    def update_dict(self, W, A, B):
        """Update dictionary matrix W using new aggregate matrices A and B.

        Args:
            W (ndarry): dictionary matrix with dimensions:
                data_dim (d) x topics (r)
            A (ndarry): aggregate matrix with dimensions:
                topics (r) x topics(r)
            B (ndarry): aggregate matrix with dimensions:
                topics (r) x data_dim(d)

        Returns:
            W1 (ndarry): updated dictionary matrix with dimensions:
                data_dim (d) x topics (r)
        """
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop
        d, r = np.shape(W)
        W1 = W.copy()

        # ****
        for j in np.arange(r):
            W1[:, j] = W1[:, j] - (1 / (A[j, j] + 1)) * ((W1 @ A[:, j]) - B.T[:, j])
            W1[:, j] = np.maximum(W1[:, j], np.zeros(shape=(d,)))
            W1[:, j] = (1 / np.maximum(1, LA.norm(W1[:, j]))) * W1[:, j]

        return W1

    def step(self, X, A, B, W, t):
        """Perform a single iteration of the online NMF algorithm.

        Note: H (ndarry): code matrix with dimensions:
            topics (r) x samples(n)

        Args:
            X (ndarry): data matrix with dimensions:
                data_dim (d) x samples (n)
            A (ndarry): aggregate matrix with dimensions:
                topics (r) x topics(r)
            B (ndarry): aggregate matrix with dimensions:
                topics (r) x data_dim(d)
            W (ndarry): dictionary matrix with dimensions:
                data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm

        Returns:
            Updated H, A, B, and W after one iteration of online NMF
            algorithm (H1, A1, B1, and W1 respectively)
        """
        # Compute H1 by coding X using dictionary W.
        H1 = self.get_code_matrix(X, W=W)

        if DEBUG:
            print(H1.shape)  # (self.num_topics, num_samples)
            print(X.shape)  # (data_dim, num_samples)
            print(B.shape)  # (self.num_topics, data_dim)

        # Update aggregate matrices A and B.
        # t = t.astype(float)
        A1 = (1 - (t ** (-self.beta))) * A + t ** (-self.beta) * H1 @ H1.T
        B1 = (1 - (t ** (-self.beta))) * B + t ** (-self.beta) * H1 @ X.T

        # Update dictionary matrix.
        W1 = self.update_dict(W, A, B)

        return H1, A1, B1, W1

    def train_dict(self, get_dicts=None):
        """Learn a dictionary W with num_topics number of columns from data.

        Learns a dictionary matrix W with num_topics number of columns based
        on a fixed stack of data matrices X from bottom to top.

        Args:
            get_dicts (list of ints): Inidices of the dictionaries to be returned.

        Returns:
            W (ndarry): dictionary matrix with dimensions:
                data_dim (d) x topics (r)
            A (ndarry): aggregate matrix with dimensions:
                topics (r) x topics(r)
            B (ndarry): aggregate matrix with dimensions:
                topics (r) x data_dim(d)
        """
        # Peek at first element of data stack to get data dimensions.
        X0 = next(iter(self.data_stack))
        data_dim = X0.shape[0]

        # Initialize matrices.
        if self.initial_dict is None:
            # Initialize dictionary matrix W with random values.
            W = np.random.rand(data_dim, self.num_topics)
        else:
            W = self.initial_dict

        # Form list of dictionaries to be returned.
        if get_dicts is not None:
            seq_dict = []
            if 0 in get_dicts:
                seq_dict.append(W)
        else:
            seq_dict = None

        # Initialize aggregate matrices A, B with zeros.
        A = np.zeros((self.num_topics, self.num_topics))
        B = np.zeros((self.num_topics, data_dim))

        for j, X in tqdm(enumerate(self.data_stack)):
            # Get number of data samples in X slice.
            # The X may have different numbers of samples.
            num_samples = X.shape[1]

            for i in tqdm(range(1, self.iterations), leave=False):
                # Randomly subsample data slice.
                idx = np.random.randint(num_samples, size=self.batch_size)
                X_batch = X[:, idx]
                if not isinstance(X, np.ndarray):
                    X_batch.toarray()

                # Iteratively update W, A, and B using batches of X.
                H, A, B, W = self.step(X_batch, A, B, W, i + (j * self.iterations))

            if get_dicts is not None and j + 1 in get_dicts:
                seq_dict.append(W)

        return W, A, B, seq_dict
