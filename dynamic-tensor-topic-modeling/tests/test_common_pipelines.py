import os

import pytest
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from covid19.twitter.common_pipelines import (
    get_bag_of_words_per_file,
    get_tweet_text_pipeline,
)


@pytest.fixture()
def filepaths(datadir):
    filepaths = [
        os.path.join(datadir, "test_tweets{}.jsonl.gz".format(i)) for i in range(3)
    ]
    return filepaths


def test_text_from_single_file(filepaths):

    tweet_text_pipeline = get_tweet_text_pipeline(filepaths[0], include_retweets=False)
    assert list(tweet_text_pipeline) == [
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
        "foo",
        "Hi!",
    ]

    # Include retweets.
    tweet_text_pipeline = get_tweet_text_pipeline(filepaths[0], include_retweets=True)
    assert list(tweet_text_pipeline) == [
        "Hello my friend, how are you?",
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
        "My dog ate my homework. Sorry it is late.",
        "foo",
        "foo",
        "Hi!",
        "Hi!",
    ]


def test_text_froms_files(filepaths):

    tweet_text_pipeline = get_tweet_text_pipeline(*filepaths, include_retweets=False)
    assert list(tweet_text_pipeline) == [
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
        "foo",
        "Hi!",
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
        "foo",
        "Hi!",
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
        "foo",
        "Hi!",
    ]


def test_text_froms_files_subsampled(filepaths):
    """Test number of elements when subsampling."""
    num_samples = 5

    tweet_text_pipeline = get_tweet_text_pipeline(
        *filepaths, include_retweets=False, num_samples=num_samples
    )
    assert len(list(tweet_text_pipeline)) == num_samples

    tweet_text_pipeline = get_tweet_text_pipeline(
        *filepaths, include_retweets=True, num_samples=num_samples
    )
    assert len(list(tweet_text_pipeline)) == num_samples


def test_bag_of_words(filepaths):
    """Test dimensions of elements in get_bag_of_words_per_file."""
    samples_per_file = 2
    tokenizer = TweetTokenizer(preserve_case=False)
    vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize)
    vectorizer.fit(get_tweet_text_pipeline(*filepaths, include_retweets=False))
    bags_of_words_per_file = get_bag_of_words_per_file(
        filepaths, vectorizer, include_retweets=False, samples_per_file=samples_per_file
    )

    assert len(bags_of_words_per_file) == len(filepaths)

    assert next(iter(bags_of_words_per_file)).shape == (
        len(vectorizer.vocabulary_),
        samples_per_file,
    )
