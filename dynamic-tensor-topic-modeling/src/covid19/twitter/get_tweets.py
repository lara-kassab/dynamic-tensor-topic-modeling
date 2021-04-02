"""Iterables for streaming tweets from files."""

from covid19.fileio import LinesFromGzipFiles
from covid19.twitter import clean_text, tweet_info
from covid19.utils import Pipeline, negate


class TweetsFromFiles(Pipeline):
    """Iterate over tweets from the given .jsonl.gz files.

    Args:
        filepaths (iterable of str): Paths to gzip files of tweets.

    Yields:
        dict: Tweet object.

    Notes:
        See `covid19.utils.Pipeline` for more details.
    """

    def __init__(self, *filepaths):
        lines = LinesFromGzipFiles(filepaths)
        steps = [(map, tweet_info.tweet_from_json_line)]
        super().__init__(lines, steps)


def text_from_tweets(tweets, include_retweets=True):
    """Return a generator of text of tweets from the indicated range.

    Args:
        tweets (iterable of dict): Iterable of tweets. Will not be consumed.
        include_retweets: True if retweets should be included, False otherwise.

    Returns:
        iterable of str: Tweet body text generator.
    """

    # Only keep English tweets.
    tweets = filter(tweet_info.is_english, tweets)

    if not include_retweets:
        # Remove retweets.
        tweets = filter(negate(tweet_info.is_retweet), tweets)

    # Preprocess tweets.
    # Only keep text from tweets, remove links and user tags.
    tweet_texts = map(tweet_info.get_full_text, tweets)
    tweet_texts = map(clean_text.remove_links, tweet_texts)
    tweet_texts = map(clean_text.remove_user_tags, tweet_texts)

    return tweet_texts
