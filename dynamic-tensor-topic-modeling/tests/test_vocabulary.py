import os

from covid19.twitter import common_pipelines


def test_get_tweet_text(datadir):
    filepath = os.path.join(datadir, "test_tweets.jsonl.gz")
    tweet_text = common_pipelines.get_tweet_text_pipeline(
        filepath, include_retweets=True
    )
    assert list(tweet_text) == [
        "Hello my friend, how are you?",
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
        "My dog ate my homework. Sorry it is late.",
    ]

    tweet_text = common_pipelines.get_tweet_text_pipeline(
        filepath, include_retweets=False
    )
    assert list(tweet_text) == [
        "Hello my friend, how are you?",
        "My dog ate my homework. Sorry it is late.",
    ]
