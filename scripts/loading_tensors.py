import os
import pickle

from build_retweet_tensors import build_corpus, build_corpus_random, build_tensor_full


def loading_tensors(main_directory, tweets_filename, k, opt, skip=0):

    """ Given a pickle file of all tweet objects, save the data tensor and extracted features.
        Args:
            main_directory (str): directory containing pickle file of tweet objects, and where files will be saved
            tweets_filename (str): name of the pickle file containing all tweet objects
            k (int): number of documents per timeframe
            opt (0 or 1): 0 to run 'build_corpus_random' (for random tweets), and 1 to run 'build_corpus' (for top tweets)
            skip (int): number of days (or time frames) to skip at the beginning
    """

    if opt == 0:
        pathname = os.path.join(main_directory, tweets_filename)
        corpus = build_corpus_random(pathname, k, skip)
    elif opt == 1:
        pathname = os.path.join(main_directory, tweets_filename)
        corpus = build_corpus(pathname, k, skip)

    corpus_directory = os.path.join(main_directory, "data" + tweets_filename)
    with open(corpus_directory, "wb") as f:
        pickle.dump(corpus, f)

    [features, X] = build_tensor_full(corpus, k, max_features=5000)

    features_directory = os.path.join(main_directory, "data_tensor" + tweets_filename)
    with open(features_directory, "wb") as f:
        pickle.dump([features, X], f)

    return
