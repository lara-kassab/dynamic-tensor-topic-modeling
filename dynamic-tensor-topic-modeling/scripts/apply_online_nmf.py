"""Apply online NMF to tweets with a loaded vocabulary to learn a dictionary.

The loaded vocabulary will typically be the combined vocabulary for the tweet
files considered.
"""

import itertools
import os
from functools import partial

import cloudpickle  # Need cloudpickle to pickle TfidfVectorizer.
import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Local imports.
import reproducibility
from config import data_dir
from covid19.online_nmf import nmf_utils, onmf
from covid19.text import stopwords
from covid19.twitter import common_pipelines, file_mgmt
from covid19.utils import Pipeline

# TODO: Generalize to allow time slices other than per file.
# TODO: Use mlflow to manage "experiments" or runs of this file.

# Parameters.
min_term_freq = 0.0005
samples_per_file = 1000
data_range_low = "2020-02-01-00"
data_range_high = "2020-02-01-23"
include_retweets = False
max_ngram_len = 2
# Parameters for online NMF.
nmf_kwargs = {
    "num_topics": 10,
    "regularization_param": 0,
    "iterations": 100,
    "batch_size": 50,
    "ini_dict": None,
}

nmf_dir = os.path.join(data_dir, "nmf")

tokenizer = TweetTokenizer(preserve_case=False)
vectorizer = TfidfVectorizer(
    analyzer="word",
    tokenizer=tokenizer.tokenize,
    min_df=min_term_freq,
    ngram_range=(1, int(max_ngram_len)),
    stop_words=stopwords.stopwords,
)

# Get files in the range indicated by the saved vocabulary.
filepaths = list(
    file_mgmt.files_in_range(
        data_dir, data_range_low=data_range_low, data_range_high=data_range_high
    )
)

# Get text of tweets.
tweet_pipeline = Pipeline(filepaths, precompute_len=True)
tweet_pipeline.add_map(
    partial(
        common_pipelines.get_tweet_text_pipeline,
        include_retweets=include_retweets,
        num_samples=samples_per_file,
    )
)

# Fit vectorizer with all tweets.
vectorizer.fit(itertools.chain.from_iterable(tqdm(tweet_pipeline)))

# Transform text to bag of words using vectorizer.
tweet_pipeline.add_map(vectorizer.transform)
# Take transpose of each bag of words matrix to match input required by
# onmf.py.
tweet_pipeline.add_map(csr_matrix.transpose)

# Online NMF. Parameters are defined at the top of the file.
nmf = onmf.Online_NMF(tweet_pipeline, **nmf_kwargs)

W, A, B = nmf.train_dict()

# Save NMF result.
nmf_dict = {
    "W": W,
    "A": A,
    "B": B,
    "min_term_freq": min_term_freq,
    "samples_per_file": samples_per_file,
    # "vocab_filename": vocab_filename,
    "vectorizer": vectorizer,
    "nmf_kwargs": nmf_kwargs,
    "data_range_low": data_range_low,
    "data_range_high": data_range_high,
    "include_retweets": include_retweets,
}

# Form filename.
nmf_params = itertools.chain.from_iterable(nmf_kwargs.items())
nmf_params = list(map(str, nmf_params))
filename = "_".join(
    [
        "nmf_dictionary",
        data_range_low,
        "to",
        data_range_high,
        "samples_per_file",
        str(samples_per_file),
        "min_term_freq",
        str(min_term_freq),
        "include_retweets",
        str(include_retweets),
        "max_ngram_len",
        str(max_ngram_len),
        *nmf_params,
    ]
)

# Save and log output.
output_dir = reproducibility.log_output(filename, results_dir=nmf_dir)
output_filepath = os.path.join(output_dir, filename)
with open(output_filepath + ".pickle", "wb") as file:
    cloudpickle.dump(nmf_dict, file)

# Print the most representative words for each topic.
num_keywords_to_print = 10
idx_to_word = np.array(vectorizer.get_feature_names())
nmf_utils.topic_keywords(num_keywords_to_print, W, idx_to_word)

print(filename)
