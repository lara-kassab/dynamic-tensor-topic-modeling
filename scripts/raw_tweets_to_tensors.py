import os
import pickle

from build_retweet_tensors import build_corpus, build_corpus_random, build_tensor_full

from config import data_dir

"""
Given a pickle file of all tweet objects, save the data tensor and extracted features.
Args:
    data_dir (str): directory containing pickle file of tweet objects
    tweets_filename (str): name of the pickle file containing all tweet objects
    k (int): number of documents per time slice
    opt (boolean): 0 for random tweets, and 1 for top tweets
    skip (int): number of days (or time slices) to skip at the beginning
"""

k = 1000
opt = 1  # 1 = ordered tweets, 0 = random.
skip = 0
tweets_filename = "tweet_tensor"  # Fill


# Get list of tweet texts.
if opt == 0:
    pathname = os.path.join(data_dir, tweets_filename)
    corpus = build_corpus_random(pathname, k, skip)
elif opt == 1:
    pathname = os.path.join(data_dir, tweets_filename)
    corpus = build_corpus(pathname, k, skip)

corpus_directory = os.path.join(data_dir, "data" + tweets_filename)
with open(corpus_directory, "wb") as f:
    pickle.dump(corpus, f)

[features, X] = build_tensor_full(corpus, k, max_features=5000)

features_directory = os.path.join(data_dir, "data_tensor" + tweets_filename)
with open(features_directory, "wb") as f:
    pickle.dump([features, X], f)
