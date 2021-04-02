# (setq python-shell-interpreter "./venv/bin/python")

import numpy as np
# import progressbar
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac
from sklearn.decomposition import SparseCoder
import time
from tqdm import trange
import os
import psutil

DEBUG = False


class Online_CPDL():

    def __init__(self,
                 X, n_components=100,
                 iterations=500,
                 sub_iterations=10,
                 batch_size=20,
                 ini_CPdict=None,
                 ini_loading=None,
                 history=0,
                 ini_A=None,
                 ini_B=None,
                 alpha=None,
                 beta=None,
                 subsample=True):
        '''
        Online CP Dictionary Learning algorithm
        X: data tensor (n+1 -dimensional) with shape I_1 * I_2 * ... * I_n * I_n+1
        Last node considered as the "batch mode"
        Seeks to find CP dictionary D = [A1, A2, ... , A_R], R=n_components, Ai = rank 1 tensor
        Such that each slice X[:,:,..,:, j] \approx <D, c> for some c
        n_components (int) = r = number of rank-1 CP factors
        iter (int): number of iterations where each iteration is a call to step(...)
        batch_size (int): number random of columns of X that will be sampled during each iteration
        '''
        self.X = X
        self.n_modes = X.ndim - 1  ### -1 for discounting the last "batch_size" mode
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.At = None  # code_covariance matrix to be learned

        if ini_CPdict is not None:
            self.CPdict = ini_CPdict
        else:
            self.CPdict = self.initialize_CPdict()

        if ini_loading is None:
            self.loading = self.initialize_loading()
        else:
            self.loading = ini_loading

        if ini_A is not None:
            self.ini_A = ini_A
        else:
            self.ini_A = np.zeros(shape=(n_components, n_components))
        # print('A', self.ini_A)

        if ini_A is not None:
            self.ini_B = ini_B
        else:
            Y = X.take(indices=0, axis=-1)
            self.ini_B = np.zeros(shape=(len(Y.reshape(-1, 1)), n_components))

        self.history = history
        self.alpha = alpha
        # print('???????alpha', alpha)
        self.beta = beta
        self.code = np.zeros(shape=(n_components, X.shape[-1]))  ### X.shape[-1] = batch_size
        self.subsample = subsample

    def initialize_loading(self):
        ### loading = python dict of [U1, U2, \cdots, Un], each Ui is I_i x R array
        loading = {}
        for i in np.arange(self.n_modes):  # n_modes = X.ndim -1 where -1 for the last `batch mode'
            loading.update({'U' + str(i): np.random.rand(self.X.shape[i], self.n_components)})
        return loading

    def initialize_CPdict(self):
        ### CPdict = python dict of [A1, A2, \cdots, AR], R=n_components, each Ai is a rank-1 tensor
        CPdict = {}
        for i in np.arange(self.n_components):
            CPdict.update({'A' + str(i): np.zeros(shape=self.X.shape[:-1])})
            ### Exclude last "batch size" mode
        return CPdict

    def out(self, loading, drop_last_mode=False):
        ### given loading, take outer product of respected columns to get CPdict
        ### Use drop_last_mode for ALS
        CPdict = {}
        for i in np.arange(self.n_components):
            A = np.array([1])
            if drop_last_mode:
                n_modes_multiplied = len(loading.keys()) - 1
            else:
                n_modes_multiplied = len(loading.keys())  # also equals self.X_dim - 1
            for j in np.arange(n_modes_multiplied):
                loading_factor = loading.get('U' + str(j))  ### I_i X n_components matrix
                # print('loading_factor', loading_factor)
                A = np.multiply.outer(A, loading_factor[:, i])
            A = A[0]
            CPdict.update({'A' + str(i): A})
        return CPdict


    def sparse_code_tensor(self, X, CPdict, H0=None, r=None, sparsity=0, sub_iter=[10], stopping_grad_ratio=0.01):
        '''
        Given data tensor X and CP dictionary CPdict, find sparse code c such that
        X \approx <CPdict, c>
        args:
            X (numpy array): data tensor with dimensions: (I1) x (I2) x ... x (In) x (In+1)
            CPdict (numpy dictionary): [A1, ... AR], R=n_componetns, Ai = rank 1 tensor
            H0 (numpy array): initial code matrix with dimensions: (R) x (In+1)
            r (float): the new code matrix H is found in the ball of radius r centered at H0
        returns:
            c (numpy array): sparse code with dimensions: topics (n_components) x samples (In+1)
        method:
            matricise X into (I1 ... In) x (In+1) and Ai's and use the usual sparse coding
        '''

        ### make a dictionary matrix from CPdict

        W = np.zeros(shape=(len(X.reshape(-1, X.shape[-1])), self.n_components))

        for j in np.arange(self.n_components):
            W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', CPdict.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H


        c = update_code_within_radius(X.reshape(-1, X.shape[-1]), W, H0=H0, r=r, alpha=sparsity,
                                        sub_iter=sub_iter, stopping_grad_ratio=stopping_grad_ratio,
                                        subsample_ratio=None, nonnegativity=True).T

        """
        if self.alpha is None and sparsity is not None:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=2, transform_algorithm='lasso_lars', positive_code=True)
        elif sparsity is not None:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=sparsity, transform_algorithm='lasso_lars', positive_code=True)
        else:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=self.alpha, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter.
        c = coder.transform(X.reshape(-1, X.shape[-1]).T)
        """


        # transpose H before returning to undo the preceding transpose on X
        return c

    def get_A_U(self, A, loading, j):
        ### Compute (n_componetns) x (n_components) intermediate aggregation matrix
        A_U = A
        for k in np.arange(self.n_modes):
            if k != j:
                U = loading.get('U' + str(k))  ### I_{k} x n_components loading matrix
                U = U.T @ U  ### (n_componetns) x (n_components) matrix
                A_U = A_U * U
        return A_U

    def get_B_U(self, B, loading, j):
        ### Compute (n_components) x (I_j) intermediate aggregation matrix
        ### B has size (I_1 * I_2 * ... * I_n) x (n_componetns)

        B_U = np.zeros(shape=(self.n_components, self.X.shape[j]))
        for r in np.arange(self.n_components):
            B_r = B[:, r].reshape(self.X.shape[:-1])
            for i in np.arange(self.n_modes):
                if i != j:
                    U = loading.get('U' + str(i))  ### I_{k} x n_components loading matrix
                    B_r = tl.tenalg.mode_dot(B_r, U[:, r], i)
                    B_r = np.expand_dims(B_r, axis=i)  ## mode_i product above kills axis i
            B_U[r, :] = B_r.reshape(self.X.shape[j])
        return B_U

    def update_dict(self, loading, A, B, r=None, nonnegativity=True, sub_iter=1):
        '''
        Updates loading = [U1, .. , Un] using new aggregate matrices A and B
        args:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (R)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(R)
            B (numpy array): aggregate matrix with dimensions: (I1 ... In) x topics (R)
            r (float): search radius
          returns:
            loading = [U1', .. , Un'] (numpy dict): each Ui has shape I_i x R
        '''
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop

        for j in np.arange(self.n_modes):
            A_U = self.get_A_U(A, loading, j)  ### intermediate aggregate matrices updated for each j
            B_U = self.get_B_U(B, loading, j)  ### intermediate aggregate matrices updated for each j
            W1 = loading.get('U' + str(j)).copy()
            for k in np.arange(self.n_components):
                for i in np.arange(sub_iter):
                    # W1[:,k] = W1[:,k] - (1/W1[k, k])*(np.dot(W1, A[:,k]) - B.T[:,k])
                    grad = (np.dot(W1, A_U[:, k]) - B_U[k, :])
                    grad_norm = np.linalg.norm(grad, 2)
                    step_size = (1 / (((i + 1) ** (1)) * (A_U[k, k] + 1)))

                    if r is not None:  # usual sparse coding without radius restriction
                        d = step_size * grad_norm
                        step_size = (r / max(r, d)) * step_size

                    W1[:, k] = W1[:, k] - step_size * grad

                    if nonnegativity:
                        W1[:, k] = np.maximum(W1[:, k], np.zeros(shape=(W1.shape[0],)))

                    W1[:, k] = (1 / np.maximum(1, LA.norm(W1[:, k]))) * W1[:, k] # normalization

            loading.update({'U' + str(j): W1})  ### Lindeberg replacement trick here

        return loading

    def step(self, X, A, B, loading, t, search_radius_const=None, beta=None, sub_iter=1):
        '''
        Performs a single iteration of the online NMF algorithm from
        Han's Markov paper.
        Note: H (numpy array): code matrix with dimensions: topics (r) x samples(n)
        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim(d)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm
            search_radius_const (float)
            beta (float): search radius = function(t, search_radius_const, beta)
        returns:
            Updated versions of H, A, B, and W after one iteration of the online CPDL
            algorithm (H1, A1, B1, and W1 respectively)
        '''

        # Compute H1 by sparse coding X using dictionary W
        CPdict = self.out(loading)
        H1 = self.sparse_code_tensor(X, CPdict) # no need to restrict search radius

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        t = float(t)
        if self.beta == None:
            beta = 1
        else:
            beta = self.beta
        A1 = (1 - (t ** (-beta))) * A + t ** (-beta) * np.dot(H1.T, H1)
        B1 = (1 - (t ** (-beta))) * B + t ** (-beta) * np.dot(X.reshape(-1, X.shape[-1]), H1)

        # Update dictionary matrices
        if search_radius_const is None:
                search_radius_const = 10000

        search_radius = search_radius_const * (float(t+1))**(-beta)/np.log(float(t+2))
        loading1 = self.update_dict(loading, A1, B1, r=search_radius, sub_iter=sub_iter)
        self.history = t + 1
        # print('history=', self.history)
        return H1, A1, B1, loading1

    def train_dict(self,
                   sub_iter_dict_update=1,
                   mode_2be_subsampled=None,
                   output_results=True,
                   save_folder=None,
                   if_compute_recons_error=False,
                   if_print_memory_usage=False):
        '''
        Given data tensor X, learn loading matrices L=[U0, U1, \cdots, Un-1] that gives CPdict = out(L).
        '''

        A = self.ini_A
        B = self.ini_B
        loading = self.loading
        code = self.code
        t0 = self.history
        X = self.X
        if mode_2be_subsampled is not None:
            # Make the mode to be subsampled the last mode of the tensor
            X = np.swapaxes(X, mode_2be_subsampled, -1)
        else:
            mode_2be_subsampled = -1

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0

        for i in trange(1, self.iterations):
            if if_print_memory_usage:
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                print('memory use:', memoryUse)


            start = time.time()
            X_batch = X
            idx = np.arange(X.shape[-1])
            # randomly choose batch_size number of "columns" to sample
            # initializing the "batch" of X, which are the subset
            # of columns from X_unfold that were randomly chosen above
            if self.subsample:
                idx = np.random.randint(X.shape[-1], size=self.batch_size)
                X_batch = X_batch.take(indices=idx, axis=-1)

            # iteratively update "loading" using batches of X, along with
            # iteratively updating values of A and B
            # print('X.shape before training step', self.X.shape)

            H, A, B, loading = self.step(X_batch, A, B, loading, t0 + i, beta=self.beta)
            end = time.time()
            elapsed_time += end - start

            code[:, idx] += H.T
            # print('dictionary=', W)
            # print('code=', H)
            # plt.matshow(H)
            if if_compute_recons_error:
                # Last mode of X (currently swapped) is regarded as the batch mode
                # U_last.shape[0] = self.batch_size
                loading_new = loading.copy()
                Code = self.sparse_code_tensor(X, self.out(loading))
                U_new = Code.reshape(X.shape[-1], self.n_components)
                loading_new.update({'U' + str(X.ndim - 1): U_new})
                CPdict_new = self.out(loading_new)
                recons = np.zeros(X.shape)
                for j in np.arange(len(loading_new.keys())):
                    recons += CPdict_new.get('A' + str(j))
                error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                print('!!! Reconstruction error at iteration %i = %f.3' % (i, error))

        if self.subsample and mode_2be_subsampled != -1:
            print('!!! mode_2be_subsampled', mode_2be_subsampled)
            # Swap back the loading matrices swapped before training
            U_swapped = loading.get(
                'U' + str(mode_2be_subsampled)).copy()  # This loading matrix needs to go to the last mode
            U_subsampled = loading.get('U' + str(self.n_modes)).copy()
            loading.update({'U' + str(self.n_modes): U_swapped})
            loading.update({'U' + str(mode_2be_subsampled): U_subsampled})

        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        # result_dict.update({'CODE_COV_MX': A})
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': self.iterations})
        result_dict.update({'n_components': self.n_components})
        result_dict.update({'data_shape': self.X.shape})

        if save_folder is not None:
            np.save(save_folder + "/OCPDL_result_", result_dict)

            #  progress status
            # print('Current iteration %i out of %i' % (i, self.iterations))
        self.CPdict = self.out(loading)  ### Rank 1 CP factors
        self.loading = loading
        self.At = A
        self.code = code

        if output_results:
            return result_dict
        else:
            return loading, A, B, code

    def inner_product_loading_code(self, loading, code):
        ### loading = [U1, .. , Un], Ui has shape Ii x R
        ### code = shape R x batch_size
        ### compute the reconstructed tensor of size I1 x I2 x .. x In x batch_size
        ### from the given loading = [U1, .. , Un] and code = [c1, .., c_batch_size], ci = (R x 1)
        recons = np.zeros(shape=self.X.shape[:-1])
        recons = np.expand_dims(recons, axis=-1)  ### Now shape I1 x I2 x .. x In x 1
        CPdict = self.out(loading)
        for i in np.arange(code.shape[1]):
            A = np.zeros(self.X.shape[:-1])
            for j in np.arange(len(loading.keys())):
                A = A + CPdict.get('A' + str(j)) * code[j, i]
            A = np.expand_dims(A, axis=-1)
            recons = np.concatenate((recons, A), axis=-1)
        recons = np.delete(recons, obj=0, axis=-1)
        # print('recons.shape', recons.shape)
        return recons

    def compute_reconstruction_error(self, X, loading, is_X_full_tensor=False):
        ### X data tensor, loading = loading matrices
        ### Find sparse code and compute reconstruction error
        ### If X is full tensor,
        c = self.sparse_code_tensor(X, self.out(loading))
        # print('X.shape', X.shape)
        # print('c.shape', c.shape)
        recons = self.inner_product_loading_code(loading, c.T)
        error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
        return error

    def ALS(self,
            iter=100,
            ini_loading=None,
            beta=None,
            if_compute_recons_error=False,
            save_folder='Output_files',
            search_radius_const=1000,
            output_results=False):
        '''
        Given data tensor X and initial loading matrices W_ini, find loading matrices W by Alternating Least Squares
        '''

        X = self.X

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', ini_loading.shape, '\n')

        n_modes = len(X.shape)
        if ini_loading is not None:
            loading = ini_loading
        else:
            loading = {}
            for i in np.arange(X.ndim):
                loading.update({'U' + str(i): np.random.rand(X.shape[i], self.n_components)})

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0

        for step in trange(int(iter)):
            start = time.time()

            for mode in np.arange(n_modes):
                X_new = np.swapaxes(X, mode, -1)

                U = loading.get('U' + str(mode))  # loading matrix to be updated
                loading_new = loading.copy()
                loading_new.update({'U' + str(mode): loading.get('U' + str(n_modes - 1))})
                loading_new.update({'U' + str(n_modes - 1): U})

                X_new_mat = X_new.reshape(-1, X_new.shape[-1])
                CPdict = self.out(loading_new, drop_last_mode=True)

                W = np.zeros(shape=(X_new_mat.shape[0], self.n_components))
                for j in np.arange(self.n_components):
                    W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

                if beta is None:  # usual nonnegative sparse coding
                    Code = update_code_within_radius(X_new_mat, W, U.T, r=None, alpha=0)
                else:
                    if search_radius_const is None:
                        search_radius_const = 10000

                    search_radius = search_radius_const * (float(step + 1)) ** (-beta) / np.log(float(step + 2))

                    # sparse code within radius
                    Code = update_code_within_radius(X_new_mat, W, U.T, search_radius, alpha=0)

                    # print('!!!!! search_radius_const', search_radius_const)
                U_new = Code.T.reshape(U.shape)

                loading.update({'U' + str(mode): U_new})

                # print('!!! Iteration %i: %i th loading matrix updated..' % (step, mode))

            end = time.time()
            elapsed_time += end - start

            if if_compute_recons_error:
                CPdict = self.out(loading, drop_last_mode=False)
                recons = np.zeros(X.shape)
                for j in np.arange(len(loading.keys())):
                    recons += CPdict.get('A' + str(j))
                error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                print('!!! Reconstruction error at iteration %i = %f.3' % (step, error))

        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': iter})
        result_dict.update({'n_components': self.n_components})
        np.save(save_folder + "/ALS_result_", result_dict)

        if output_results:
            return result_dict
        else:
            return loading


    def MU(self,
           iter=100,
           ini_loading=None,
           if_compute_recons_error=False,
           save_folder='Output_files',
           output_results=False):
        '''
        Given data tensor X and initial loading matrices W_ini, find loading matrices W by Multiplicative Update
        Ref: Shashua, Hazan, "Non-Negative Tensor Factorization with Applications to Statistics and Computer Vision" (2005)
        '''

        X = self.X

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', ini_loading.shape, '\n')

        n_modes = len(X.shape)
        if ini_loading is not None:
            loading = ini_loading
        else:
            loading = {}
            for i in np.arange(X.ndim):
                loading.update({'U' + str(i): np.random.rand(X.shape[i], self.n_components)})

        result_dict = {}
        time_error = np.zeros(shape=[0, 2])
        elapsed_time = 0

        for step in trange(int(iter)):
            start = time.time()

            for mode in np.arange(n_modes):
                # print('!!! X.shape', X.shape)
                X_new = np.swapaxes(X, mode, -1)
                U = loading.get('U' + str(mode))  # loading matrix to be updated
                loading_new = loading.copy()
                loading_new.update({'U' + str(mode): loading.get('U' + str(n_modes - 1))})
                loading_new.update({'U' + str(n_modes - 1): U})
                # Now update the last loading matrix U = 'U' + str(n_modes - 1)) by MU
                # Matrize X along the last mode to get a NMF problem V \approx W*H, and use MU in LEE & SEUNG (1999)

                # Form dictionary matrix
                CPdict = self.out(loading_new, drop_last_mode=True)
                # print('!!! X_new.shape', X_new.shape)
                W = np.zeros(shape=(len(X_new.reshape(-1, X_new.shape[-1])), self.n_components))
                for j in np.arange(self.n_components):
                    W[:, j] = CPdict.get('A' + str(j)).reshape(-1, 1)[:, 0]

                V = X_new.reshape(-1, X_new.shape[-1])
                # print('!!! W.shape', W.shape)
                # print('!!! U.shape', U.shape)
                # print('!!! V.shape', V.shape)
                U_new = U.T * (W.T @ V) / (W.T @ W @ U.T)
                loading.update({'U' + str(mode): U_new.T})

                # print('!!! Iteration %i: %i th loading matrix updated..' % (step, mode))

            end = time.time()
            elapsed_time += end - start

            if if_compute_recons_error:
                CPdict = self.out(loading, drop_last_mode=False)
                recons = np.zeros(X.shape)
                for j in np.arange(len(loading.keys())):
                    recons += CPdict.get('A' + str(j))
                error = np.linalg.norm((X - recons).reshape(-1, 1), ord=2)
                time_error = np.append(time_error, np.array([[elapsed_time, error]]), axis=0)
                # print('!!! Reconstruction error at iteration %i = %f.3' % (step, error))

        result_dict.update({'loading': loading})
        # result_dict.update({'CPdict': self.out(loading)}) ### this is very expensive
        result_dict.update({'time_error': time_error.T})
        result_dict.update({'iter': iter})
        result_dict.update({'n_components': self.n_components})
        np.save(save_folder + "/MU_result_", result_dict)

        if output_results:
            return result_dict
        else:
            return loading


### Helper functions


def update_code_within_radius(X, W, H0=None, r=None, alpha=0, sub_iter=[10], stopping_grad_ratio=0.05, subsample_ratio=None, nonnegativity=True):
    '''
    Find \hat{H} = argmin_H ( | X - WH| + alpha|H| ) within radius r from H0
    Use row-wise projected gradient descent
    Do NOT sparsecode the whole thing and then project -- instable
    12/5/2020 Lyu

    For NTF problems, X is usually tall and thin so it is better to subsample from rows
    12/25/2020 Lyu

    Apply single round of AdaGrad for rows, stop when gradient norm is small and do not make update
    12/27/2020 Lyu
    '''

    # print('!!!! X.shape', X.shape)
    # print('!!!! W.shape', W.shape)
    # print('!!!! H0.shape', H0.shape)
    if H0 is None:
        H0 = np.random.rand(W.shape[1], X.shape[1])

    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[0])
    # print('!!! X.shape', X.shape)
    while (i < np.random.choice(sub_iter)):
        if_continue = np.ones(H0.shape[0])  # indexed by rows of H
        H1_old = H1.copy()
        b = 10
        if (subsample_ratio is not None) and (X.shape[0]>X.shape[1]):
            idx = np.random.randint(X.shape[0], size=X.shape[0]//subsample_ratio)
            A = W[idx,:].T @ W[idx,:]
            B = W[idx,:].T @ X[idx,:]
        elif (subsample_ratio is not None) and (X.shape[0]<= X.shape[1]):
            idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
            A = W[:,:].T @ W[:,:]
            B = W[:,:].T @ X[:,idx]
        else:
            A = W[:,:].T @ W[:,:]
            B = W[:,:].T @ X[:,:]

        for k in [k for k in np.arange(H0.shape[0]) if if_continue[k]>0.5]:

            grad = (np.dot(A[k, :], H1) - B[k, :] + alpha * np.ones(H0.shape[1]))
            grad_norm = np.linalg.norm(grad, 2)
            step_size = (1 / (((i + 1) ** (1)) * (A[k, k] + 1)))
            if r is not None:  # usual sparse coding without radius restriction
                d = step_size * grad_norm
                step_size = (r / max(r, d)) * step_size


            if step_size * grad_norm / np.linalg.norm(H1_old, 2) > stopping_grad_ratio:
                H1[k, :] = H1[k, :] - step_size * grad
            else:
                if_continue[k] = 1  # stop making changes when negligible
            if nonnegativity:
                H1[k,:] = np.maximum(H1[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint

        i = i + 1


    return H1
