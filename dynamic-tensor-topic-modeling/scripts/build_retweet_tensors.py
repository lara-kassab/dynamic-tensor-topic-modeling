"""Build a tensor from tweets for fixed timeframes."""

import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorly import fold

from covid19.text import stopwords

# TODO: combine build_corpus and build_corpus_random based on tweet_data.


def build_corpus(filename, k, skip=0):
    """ Build corpus from tweet pickle files.
        Args:
            filename (str): path to pickle file containing all tweet objects
            k (int): number of documents per timeframe
            skip (int): number of timeframes to skip at the beginning

        Returns:
            corpus (list): list containing all tweet texts from all pickle files in directory.
    """
    corpus = []
    retweet_counts = []
    tweet_data = pickle.load(open(filename, "rb"))
    for i in range(len(tweet_data) - skip):
        for j in range(k):
            corpus.append(tweet_data[i + skip][2][j]["full_text"])
            retweet_counts.append(tweet_data[i + skip][2][j]["retweet_count"])

    return corpus


def build_corpus_random(filename, k, skip=0):
    """ Build corpus from tweet pickle files.
        Args:
            filename (str): path to pickle file containing all tweet objects
            k (int): number of documents per timeframe
            skip (int): number of timeframes to skip at the beginning

        Returns:
            corpus (list): list containing all tweet texts from all pickle files in directory.
    """
    corpus = []
    retweet_counts = []
    tweet_data = pickle.load(open(filename, "rb"))
    for i in range(len(tweet_data) - skip):
        for j in range(k):
            corpus.append(tweet_data[i + skip][j]["full_text"])
            retweet_counts.append(tweet_data[i + skip][j]["retweet_count"])

    return corpus

def build_tensor_full(
    corpus,
    k,
    max_features=5000,
    stop_words=stopwords.stopwords,
    max_df=1.0,
    ngram_range=(1, 2),
    min_df=1,
):
    """Build a tf-idf weight tensor of dimension timeframes-by-features-by-documents.

    The dimensions of the tensor are  timeframes-by-max_features-by-k.

    Args:
        corpus (list): list of all documents
        k (int): number of documents per timeframe (used to fold the tensor)
        stop_words (list): TfidfVectorizer/CountVectorizer parameter
        max_features (int): maximum number of features (i.e. words & emojis) extracted from the documents
        max_df (int): TfidfVectorizer/CountVectorizer parameter
        min_df (int): TfidfVectorizer/CountVectorizer parameter
        ngram_range (tuple): TfidfVectorizer/CountVectorizer parameter

     Returns:
         feature_names (list): list of all features (words) extracted from documents
         X (ndarray): tf-idf weight tensor of dimension timeframes-by-features-by-documents
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=stop_words,
        max_df=max_df,
        min_df=min_df,
    )
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = np.array(dense).transpose()

    X = fold(denselist, 1, (denselist.shape[1] // k, denselist.shape[0], k))

    return feature_names, X


def build_tensor_timeframe(
    corpus,
    k,
    max_features=5000,
    stop_words=stopwords.stopwords,
    max_df=1.0,
    ngram_range=(1, 2),
    min_df=1,
):
    """Build a tf-idf weight tensor of dimension timeframes-by-features-by-documents, where the
        tf-idf weights are computed per timeframe and not across all timeframes (i.e. all documents).

    The dimensions of the tensor are  timeframes-by-max_features-by-k.

    Args:
        corpus (list): list of all documents
        k (int): number of documents per timeframe (used to fold the tensor)
        max_features (int): maximum number of features (i.e. words & emojis) extracted from the documents
        stop_words (list): TfidfVectorizer/CountVectorizer parameter
        max_df (int): TfidfVectorizer/CountVectorizer parameter
        ngram_range (tuple): TfidfVectorizer/CountVectorizer parameter
        min_df (int): TfidfVectorizer/CountVectorizer parameter

     Returns:
         feature_names (list): list of all features (words) extracted from documents
         X (ndarray): tf-idf weight tensor of dimension timeframes-by-features-by-documents

    """
    vectorizer_count = CountVectorizer(
        max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df
    )
    vectorizer_count.fit_transform(corpus)
    vocab_dict = vectorizer_count.vocabulary_

    vectorizer = TfidfVectorizer(vocabulary=vocab_dict)
    X = np.zeros([1, len(vocab_dict), k])

    for i in np.arange(0, len(corpus) - k - 1):
        corpus_slice = corpus[k * i : k * (i + 1)]
        vectors = vectorizer.fit_transform(corpus_slice)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = np.array(dense).transpose().reshape([1, len(vocab_dict), k])
        X = np.concatenate((X, denselist), axis=0)

    X = X[1:, :, :]

    return feature_names, X
