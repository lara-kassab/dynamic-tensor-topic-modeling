"""Find k most retweeted tweets from every day within a time range.

Output is saved to data_dir/top_k_daily as a pickled list of lists of k top retweeted tweets per day.
Format: [(first_filename_in_1st_day, last_filename_in_1st_day, [top_k_tweets_of_1st_day]), ... ]

Args:
    K: Number of tweets to get
    data_range_low (str): The lower end of the date and hour considered.
        Format should be YYYY-MM-DD-HH. Eg. 2020-03-01-00.
    data_range_high (str): The upper end of the date and hour considered.
        Format should be YYYY-MM-DD-HH. Eg. 2020-03-02-00.
    --not_sorted: If set, we do not sort tweets by the number of retweets. Note that --not_sorted=1 is not supported.


Examples:
    This file can be run as follows.
    The following specifically find 10 most retweeted tweets per day in the given date range.
    $ python k_most_retweeted_daily.py 10 2020-03-01-00 2020-03-01-24 --not_sorted
"""

import os
import sys
from functools import partial
from itertools import groupby

from config import data_dir
from covid19 import fileio
from covid19.twitter import file_mgmt, k_most_retweeted

if __name__ == "__main__":
    # Send command line arguments to the main function.
    args = sys.argv[1:]

    # If '--not_sorted' is present in the command line, ordered is false. In any case --not_sorted will no longer be
    # there.
    try:
        args.pop(args.index("--not_sorted"))
        ordered = False
    except ValueError:
        ordered = True

    k = int(args[0])
    data_range_low = "2020-01-05-00"  # first Sunday in 2020
    data_range_high = None

    if len(args) > 1:
        data_range_low = args[1]
    if len(args) > 2:
        data_range_high = args[2]

    filepaths = sorted(
        file_mgmt.files_in_range(data_dir, data_range_low, data_range_high)
    )

    top_k = []
    for day_number, paths_from_one_day in groupby(
        filepaths, partial(file_mgmt.day_from_filename, start_date_hour=data_range_low)
    ):
        this_day_paths = list(paths_from_one_day)
        top_k.append(
            (
                this_day_paths[0],
                this_day_paths[-1],
                k_most_retweeted.find_most_retweeted(k, this_day_paths, ordered),
            )
        )

    filename = "_".join([str(k), "tweets_from", data_range_low, "to", data_range_high])
    fileio.save_in_subdir(top_k, filename, data_dir)
