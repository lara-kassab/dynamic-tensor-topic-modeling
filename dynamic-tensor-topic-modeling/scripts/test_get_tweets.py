
import covid19.twitter.get_tweets
# import covid19.twitter.common_pipelines.py
from covid19.utils import Pipeline as Pipeline
# from covid19.utils import add_map as add_map
from covid19.twitter import get_tweets as get_tweets

filepaths = ["Data/winter_olympics.txt"]

tweets = get_tweets.TweetsFromFiles(*filepaths)

# tweet_files_pipeline = Pipeline(filepaths, precompute_len=True)
# tweet_files_pipeline.add_map(get_tweets.TweetsFromFiles)

