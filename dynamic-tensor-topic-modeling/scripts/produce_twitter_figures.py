"""Produce figures for "On Nonnegative Matrix and Tensor Decompositions for COVID-19 TwitterDynamics"

Data must first be collected following directions from
https://github.com/echen102/COVID-19-TweetIDs

Specify the directory where the data is stored in a file called `_local_config.py`.
Example contents of _local_config.py:
    data_dir = "/local/path/to/COVID-19-TweetIDs"
    results_dir = "/local/path/to/results_dir"

Usage:
    $ python produce_paper_figures.py
"""

import os

# Parameters.
start_datetime = "2020-02-01-00"
end_datetime = "2020-05-01-00"
tweets_per_day = 1000

# Get top daily retweeted tweets saved in the data dir with
os.system(
    "python k_most_retweeted_daily.py {} {} {}".format(
        tweets_per_day, start_datetime, end_datetime
    )
)

# Form tweet data into tensor.
os.system("python raw_tweets_to_tensors.py")

# Reproduce NMF experiments.
os.system("python Fixed_NMF.py")

# Reproduce ONMF experiments.
os.system("python apply_ONMF_to_tensor_data.py")

# Reproduce NCPD experiments.
os.system("python run_NNCPD.py")

# Reproduce ONCPD experiments.
os.system("python run_OnlineNCPD_generate_figures.py")
