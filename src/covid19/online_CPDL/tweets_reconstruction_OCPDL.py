import itertools
import os
import pickle
import re
from collections import defaultdict
from time import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas
import psutil
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)

from covid19 import plotting, utils
from covid19.online_CPDL.ocpdl import Online_CPDL

DEBUG = False


class Tweets_Reconstructor_OCPDL:
    # Use Online CP Dictionary Learning for patch-based image processing
    def __init__(
        self,
        path,
        n_components=100,  # number of dictionary elements -- rank
        iterations=50,  # number of iterations for the ONTF algorithm
        sub_iterations=20,  # number of i.i.d. subsampling for each iteration of ONTF
        batch_size=20,  # number of patches used in i.i.d. subsampling
        num_patches=1000,  # number of patches that ONTF algorithm learns from at each iteration
        sub_num_patches=10000,  # number of patches to optimize H after training W
        segment_length=20,  # number of video frames in a single video patch
        unfold_words_tweets=False,
        if_sample_from_tweets_mode=True,
        alpha=1,
    ):
        """
        batch_size = number of patches used for training dictionaries per ONTF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        """
        self.path = path
        self.n_components = n_components
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.num_patches = num_patches
        self.sub_num_patches = sub_num_patches
        self.alpha = alpha
        self.batch_size = batch_size
        self.unfold_words_tweets = unfold_words_tweets
        self.if_sample_from_tweets_mode = if_sample_from_tweets_mode
        self.segment_length = segment_length
        self.W = np.zeros(
            shape=(1, n_components)
        )  # Will be re-initialized by ocpdl.py with the correct shape
        self.code = np.zeros(shape=(n_components, iterations * batch_size))
        self.sequential_dict = {}

        dict = pickle.load(open(self.path, "rb"))
        self.X_words = dict[0]  # list of words
        # self.X_retweetcounts = dict[1]  # retweet counts
        self.data = (
            dict[1] * 10000
        )  # [timeframes, words, tweets]  # use dict[2] for old tweet data
        # scale by 100 since the original tfidf weigts are too small -- result of learning is noisy
        print("data_shape ([timeframes, words, tweets])", self.data.shape)

        self.frameCount = self.data.shape[0]
        self.num_words = self.data.shape[1]
        self.num_tweets = self.data.shape[2]

    def extract_random_patches(
        self, data=None, starting_time=None, if_sample_from_tweets_mode=True
    ):
        """
        Extract 'num_patches' many random patches of given size
        Two tensor data types depending on how to unfold (seg_length x num_words x num_tweets) short time-series patches:
            unfold_words_tweets : (seg_length, num_words * num_tweets, 1 , 1)
            else: (seg_length, num_words,  num_tweets, 1)
            last mode = batch mode
        """
        kt = self.frameCount
        ell = self.segment_length
        num_patches = self.num_patches

        if not if_sample_from_tweets_mode:
            if self.unfold_words_tweets:
                X = np.zeros(shape=(ell, self.num_words * self.num_tweets, 1, 1))
            else:
                X = np.zeros(shape=(ell, self.num_words, self.num_tweets, 1))

            for i in np.arange(num_patches):
                if starting_time is None:
                    a = np.random.choice(
                        kt - ell
                    )  # starting time of the short time-series patch
                else:
                    a = starting_time

                if data is None:
                    Y = self.data[a : a + ell, :, :]  # ell by num_words by num_tweets
                else:
                    Y = data[a : a + ell, :, :]  # ell by num_words by num_tweets
                # print('Y.shape', Y.shape)

                if self.unfold_words_tweets:
                    Y = Y.reshape(ell, self.num_words * self.num_tweets, 1, 1)
                else:
                    Y = Y.reshape(ell, self.num_words, self.num_tweets, 1)

                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=3)  # X is class ndarray
        else:
            # Sample patches from the tweets mode instead of the time mode
            # ell = self.segment_length plays the fole of patch size
            for i in np.arange(num_patches):
                a = np.arange(self.num_tweets)
                idx = np.random.choice(a, ell)

                if data is None:
                    Y = self.data[:, :, idx]  # ell by num_words by num_tweets
                else:
                    Y = data[:, :, idx]  # ell by num_words by num_tweets
                # print('Y.shape', Y.shape)
                Y = Y.reshape(data.shape[0], data.shape[1], ell, 1)

                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=3)  # X is class ndarray

        return X

    def out(self, loading):
        # given loading, take outer product of respected columns to get CPdict
        CPdict = {}
        for i in np.arange(self.n_components):
            A = np.array([1])
            for j in np.arange(len(loading.keys())):
                loading_factor = loading.get("U" + str(j))  # I_i X n_components matrix
                # print('loading_factor', loading_factor)
                A = np.multiply.outer(A, loading_factor[:, i])
            A = A[0]
            CPdict.update({"A" + str(i): A})
        return CPdict

    def show_array(self, arr):
        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.show()

    def train_dict(
        self,
        data=None,
        ini_dict=None,
        ini_At=None,
        ini_Bt=None,
        ini_history=None,
        iterations=None,
        sub_iterations=None,
        save_file_name=0,
        if_sample_from_tweets_mode=True,
        if_save=False,
        beta=None,
    ):
        print("training CP dictionaries from patches...")
        """
        Trains dictionary based on patches from an i.i.d. sequence of batch of patches
        CP dictionary learning
        """
        if ini_dict is not None:
            W = ini_dict  # Damn!! Don't make this None it will reset the seq_learning function
        else:
            W = None

        if ini_At is None:
            At = None
        else:
            At = ini_At

        if ini_Bt is None:
            Bt = None
        else:
            Bt = ini_Bt

        if ini_history is None:
            ini_his = 0
        else:
            ini_his = ini_history

        if iterations is None:
            iter = self.iterations
        else:
            iter = iterations

        if sub_iterations is None:
            sub_iter = self.sub_iterations
        else:
            sub_iter = sub_iterations

        for t in np.arange(iter):
            if data is None:
                X = self.extract_random_patches(
                    data=self.data,
                    if_sample_from_tweets_mode=if_sample_from_tweets_mode,
                )
            else:
                X = self.extract_random_patches(
                    data=data, if_sample_from_tweets_mode=if_sample_from_tweets_mode
                )

            # print('X.shape', X.shape)

            if t == 0:
                self.ntf = Online_CPDL(
                    X,
                    self.n_components,
                    ini_loading=W,
                    ini_A=At,
                    ini_B=Bt,
                    iterations=sub_iter,
                    batch_size=self.batch_size,
                    alpha=self.alpha,
                    history=ini_his,
                    subsample=False,
                    beta=beta,
                )
                W, At, Bt, H = self.ntf.train_dict(output_results=False)

                print("in training: ini_his = ", ini_his)

            else:
                self.ntf = Online_CPDL(
                    X,
                    self.n_components,
                    iterations=sub_iter,
                    batch_size=self.batch_size,
                    ini_loading=W,
                    ini_A=At,
                    ini_B=Bt,
                    alpha=self.alpha,
                    history=ini_his + t,
                    subsample=False,
                    beta=beta,
                )
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.ntf.train_dict(output_results=False)

                # code += H
            print("Current iteration %i out of %i" % (t, iter))
        self.W = self.ntf.loading
        # print('dict_shape:', self.W.shape)
        # print('code_shape:', self.code.shape)

        if if_save:
            np.save(
                "Tweets_dictionary/dict_learned_CPDL_" + str(save_file_name), self.W
            )
            np.save(
                "Tweets_dictionary/code_learned_CPDL_" + str(save_file_name), self.code
            )
        return W, At, Bt, H

    def train_sequential_dict(
        self,
        save_file_name=0,
        slide_window_by=10,
        refresh_history_as=None,
        beta=None,
        if_save=False,
    ):
        print("Sequentially training CP dictionaries from patches...")
        """
        Trains dictionary based on patches from a sequence of batch of patches
        slide the window only forward
        if refresh_history=True, aggregation matrices are reset every iteration
        CP dictionary learning
        """
        W = self.W
        At = None
        Bt = None
        t0 = 0
        t = 0  # If history starts at 0, then t^{-\beta} goes from 0 to 1 to very small.
        # To ensure monotonicity, start at t=1.
        seq_dict = {}

        while t0 + t * slide_window_by + self.segment_length <= self.frameCount:
            # X = self.extract_random_patches(if_sample_from_tweets_mode=False)
            X = self.extract_random_patches(
                starting_time=t0 + t * slide_window_by, if_sample_from_tweets_mode=False
            )

            # print('X', X)
            print("X.shape", X.shape)
            t1 = t0 + t * slide_window_by
            print("Current time slab starts at %i" % t1)

            if refresh_history_as is not None:
                ini_history = refresh_history_as
            else:
                ini_history = (
                    t + 1
                ) * self.iterations  # To ensure monotonicity, start at t=1.

            # data tensor still too large -- subsample from the tweets mode:

            if t == 0:
                ini_dict = None
            else:
                ini_dict = W

            print("ini_history", ini_history)

            if t > 0:
                print("U0 right before new phase", ini_dict.get("U0")[0, 0])

            W, At, Bt, H = self.train_dict(
                data=X,
                ini_dict=ini_dict,
                ini_At=At,
                ini_Bt=Bt,
                iterations=self.iterations,
                sub_iterations=2,
                ini_history=ini_history,
                if_sample_from_tweets_mode=True,
                if_save=False,
                beta=beta,
            )

            seq_dict.update({"W" + str(t): W.copy()})

            # for i in np.arange(t):
            #    print('U0[:,0]', seq_dict.get('W' + str(i)).get('U0')[0,0])

            # print out current memory usage
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB
            print("memory use:", memoryUse)

            t += 1
            # print('Current iteration %i out of %i' % (t, self.iterations))

        # self.W = self.ntf.loading
        self.sequential_dict = seq_dict

        # print('dict_shape:', self.W.shape)
        # print('code_shape:', self.code.shape)
        # np.save('Tweets_dictionary/dict_learned_CPDL_'+str(save_file_name), self.W)
        if if_save:
            np.save(
                "Tweets_dictionary/sequential_dict_learned_CPDL_" + str(save_file_name),
                seq_dict,
            )
        # np.save('Tweets_dictionary/code_learned_CPDL_'+str(save_file_name), self.code)

        return seq_dict
