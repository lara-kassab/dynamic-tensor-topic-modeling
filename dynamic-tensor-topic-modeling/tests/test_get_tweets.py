import os

from covid19.twitter.get_tweets import TweetsFromFiles


def test_tweets_from_files(datadir):
    filepaths = [
        os.path.join(datadir, "tweets1.jsonl.gz"),
        os.path.join(datadir, "tweets2.jsonl.gz"),
    ]
    tweets = TweetsFromFiles(*filepaths)
    assert list(tweets) == [
        {"full_text": "Text of tweet 0 in file tweets1.jsonl.gz"},
        {"full_text": "Text of tweet 1 in file tweets1.jsonl.gz"},
        {"full_text": "Text of tweet 0 in file tweets2.jsonl.gz"},
        {"full_text": "Text of tweet 1 in file tweets2.jsonl.gz"},
    ]

    # Make sure the iterable can be reused
    assert list(tweets) == [
        {"full_text": "Text of tweet 0 in file tweets1.jsonl.gz"},
        {"full_text": "Text of tweet 1 in file tweets1.jsonl.gz"},
        {"full_text": "Text of tweet 0 in file tweets2.jsonl.gz"},
        {"full_text": "Text of tweet 1 in file tweets2.jsonl.gz"},
    ]

    assert len(tweets) == 4


def test_tweets_from_single_file(datadir):
    tweets = TweetsFromFiles(os.path.join(datadir, "tweets1.jsonl.gz"))
    assert list(tweets) == [
        {"full_text": "Text of tweet 0 in file tweets1.jsonl.gz"},
        {"full_text": "Text of tweet 1 in file tweets1.jsonl.gz"},
    ]

    assert len(tweets) == 2
