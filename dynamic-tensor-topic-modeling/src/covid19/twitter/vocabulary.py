import glob
import itertools
import os
import pickle  # For saving the vocabulary.
import re  # For regex.
from collections import Counter
from functools import partial

from nltk.tokenize import TweetTokenizer

# Local imports.
from covid19.text import ngrams, stopwords
from covid19.twitter import common_pipelines, file_mgmt


def get_vocab_file_substring(include_retweets, max_ngram_len):
    """Get the substring with parameters for the saved data filename.

    >>> get_vocab_file_substring(False, 2)
    'vocab-retweets-False-ngrams-1-to-2'
    """
    return "-".join(
        [
            "vocab",
            "retweets",
            str(include_retweets),
            "ngrams",
            str(1),
            "to",
            str(max_ngram_len),
        ]
    )


class Vocabulary:
    """Class corresponding to a vocabulary with given parameters."""

    def __init__(self, data_dir="", include_retweets=True, max_ngram_len=2):
        """Store parameters for vocabulary instance."""
        self.data_dir = data_dir
        self.include_retweets = include_retweets
        self.max_ngram_len = max_ngram_len
        self.vocab_file_substring = get_vocab_file_substring(
            include_retweets, max_ngram_len
        )

    def build_vocabulary_for_file(self, filepath, save_vocab=False):
        """Get the vocabulary for each tweet file in filepaths.

        Args:
            filepath (str): The file containing the tweet data.
            include_retweets (bool): Whether retweets should be included.
            max_ngram_len (int): The max length of ngrams to include.
            save_vocab (bool): Whether to save the vocabulary.
        """
        # Parameters to save with vocabulary.
        vocab_dict = {
            "include_retweets": self.include_retweets,
            "stopwords": stopwords.stopwords,
            "max_ngram_len": self.max_ngram_len,
        }

        # Get text from English tweets.
        tweet_terms = common_pipelines.get_tweet_text_pipeline(
            filepath, self.include_retweets
        )

        # Tokenize tweets.
        tokenizer = TweetTokenizer(preserve_case=False)
        tweet_terms.add_map(tokenizer.tokenize)

        # Remove stopwords.
        tweet_terms.add_map(stopwords.remove_stopword_tokens)

        # Collect ngrams from the tokens for each tweet.
        tweet_terms.add_map(partial(ngrams.get_ngrams, self.max_ngram_len))

        # Flatten tokens to single list.
        terms = itertools.chain.from_iterable(tweet_terms)

        # Build vocabulary of terms and count occurances.
        term_counts = Counter(terms)

        # Save vocabulary.
        vocab_dict["term_counts"] = term_counts

        if save_vocab:
            # Pickle the vocabulary and associated data.
            filepath = re.sub(
                "-id-", "-{}-".format(self.vocab_file_substring), filepath
            )
            filepath = re.sub(".jsonl.gz", ".pickle", filepath)
            with open(filepath, "wb") as file:
                pickle.dump(vocab_dict, file)

        return vocab_dict

    def vocab_exists(self, tweets_filepath):
        """Return whether the vocab for filepath already exists.

        Args:
            filepath: filepath for data file.
        """
        date_hour = file_mgmt.extract_date_hour(tweets_filepath)
        year_month = file_mgmt.year_month_from_date_hour(date_hour)

        # Form of the corresponding vocab file.
        vocab_file_form = "{}/*{}-{}.pickle".format(
            year_month, self.vocab_file_substring, date_hour
        )
        match_vocab_file = os.path.join(self.data_dir, vocab_file_form)

        # Return True if a matching vocabulary file exists.
        if glob.glob(match_vocab_file, recursive=True):
            return True
        return False

    def vocab_does_not_exist(self, tweets_filepath):
        """Return whether the vocab for filepath does not already exist."""
        return not self.vocab_exists(tweets_filepath)
