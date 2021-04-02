import heapq

from covid19.twitter.get_tweets import TweetsFromFiles


def find_most_retweeted(k, filepaths, ordered):
    top_k = []
    for tweet in TweetsFromFiles(*filepaths):
        # only English and original tweets are counted (not retweets)
        if "retweeted_status" in tweet:
            continue
        if "lang" in tweet:
            if tweet["lang"] != "en":
                continue
        if len(top_k) >= k:
            if top_k[0][0] < tweet["retweet_count"]:
                heapq.heappop(top_k)
                # tweet id is used to compare tweets with the same retweet count
                heapq.heappush(top_k, (tweet["retweet_count"], tweet["id"], tweet))
        else:
            heapq.heappush(top_k, (tweet["retweet_count"], tweet["id"], tweet))

    if ordered:
        top_k = sorted(top_k, key=lambda x: -x[0])
    return [t[2] for t in top_k]
