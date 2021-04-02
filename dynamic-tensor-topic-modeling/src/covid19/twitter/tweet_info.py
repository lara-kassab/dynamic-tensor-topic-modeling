"""Functions for checking properties of tweets."""

import json


def tweet_from_json_line(line):
    r"""Extract a tweet dict from a line of json.

    Args:
        line (str): A json representation of a tweet.

    Returns:
        dict: The tweet represented by the json.

    >>> tweet_from_json_line('{"full_text": "..."}\n')
    {'full_text': '...'}
    """
    # TODO: Verify that tweet has 'full_text' at a minimum.
    return json.loads(line)


def get_full_text(tweet):
    """Return full_text of tweet object.

    Twitter stores the text of each tweet in tweet["full_text"].

    Args:
        tweet (dict): A tweet object represented as a dictionary.

    Returns:
        str: The body text of the tweet.

    Example:
        >>> tweet = {"full_text": "Body text.", "other_attribute": "value"}
        >>> get_full_text(tweet)
        'Body text.'
    """
    return tweet["full_text"]


def is_english(tweet):
    """Check if tweet is in English.

    Twitter indicates the language of each tweet using tweet["lang"].

    Args:
        tweet (dict): Tweet object with tweet["lang"] intact.

    Returns:
        bool: True if tweet is in English, False otherwise.

    Examples:
        >>> tweet = {"lang": "en", "full_text": "..."}
        >>> is_english(tweet)
        True

        >>> tweet = {"lang": "ru", "full_text": "..."}
        >>> is_english(tweet)
        False
    """
    return tweet["lang"] == "en"


def is_retweet(tweet):
    """Check if tweet is a retweet.

    Notes:
        This function uses the presence or absence of the "retweeted_status"
        key to determine if the tweet is a retweet. If keys have been stripped
        from the tweet, this function will not behave as expected.

        The value of tweet["retweeted_status"] is itself a tweet object
        corresponding to the tweet that was retweeted.

    Args:
        tweet (dict): Tweet object.

    Returns:
        bool: True if tweet is a retweet, False otherwise.

    Examples:
        >>> tweet1 = {"retweeted_status": {"full_text": "Original body text."},
        ...          "full_text": "..."}
        >>> is_retweet(tweet1)
        True

        >>> tweet2 = {"full_text": "..."}
        >>> is_retweet(tweet2)
        False

        If you would like to filter out all of the retweets from a stream of
        tweets you can do that like this:

        >>> tweets = [tweet1, tweet2]
        >>> from covid19.utils import negate
        >>> list(filter(negate(is_retweet), tweets))
        [{'full_text': '...'}]
    """
    return "retweeted_status" in tweet.keys()


def has_location(tweet):
    """Check if tweet includes location data.

    Args:
        tweet (dict): Tweet object with tweet["lang"] intact.

    Returns:
        bool: True if tweet includes location data, False otherwise.

    Examples:
        >>> tweet = {"place": {}, "full_text": "..."}
        >>> has_location(tweet)
        True

        >>> tweet = {"full_text": "..."}
        >>> has_location(tweet)
        False

    TODO:
        * Are there other places location data may be hiding?
    """
    return "place" in tweet.keys()
