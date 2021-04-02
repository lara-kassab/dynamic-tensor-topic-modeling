import datetime
import os
import pickle
from itertools import product

import numpy as np
from sklearn.decomposition import NMF

from config import data_dir, results_dir
from covid19 import plotting, utils

# We consider a TFIDF weight tensor of dimension timeframes-by-features-by-documents
# - features (list): list of all features (words) extracted from documents
# - retweet_counts (list): list containing the number of retweets for each corresponding tweet in corpus.
# - X (ndarray): TFIDF weight tensor of dimension timeframes-by-features-by-documents
ranks = [5, 20]
data_names = ["top_1000_daily"]
folder_names = ["top"]
num_keywords = 5

for r, (data_name, folder_name) in product(ranks, zip(data_names, folder_names)):
    np.random.seed(100)
    trial_params = "{}_{}_topics".format(data_name, r)
    print(trial_params)

    # Load data.
    if data_name == "random_1000_daily":
        data_file = os.path.join(
            data_dir,
            "data_tensor1000_random_tweets_daily_from_2020-02-01-00_to_2020-05-01-00.pickle",
        )  # Fill
    if data_name == "top_1000_daily":
        data_file = os.path.join(data_dir, "tweet_tensor.pickle")  # Fill
    with open(data_file, "rb") as file:
        [features, X] = pickle.load(file)

    # Reshape tensor into matrix.
    M = np.concatenate(X, 1)

    # Apply NMF.
    nmf = NMF(n_components=r, init="nndsvd", random_state=100)
    W = nmf.fit_transform(M)  # Dictionary.
    H = nmf.components_  # Topic representations.

    # Condense topic representations.
    topics_freqs = []
    for i, topic in enumerate(W.T):
        topics_freqs.append(
            {features[i]: topic[i] for i in reversed(topic.argsort()[-20:])}
        )
    topics_freqs = utils.condense_topic_keywords(topics_freqs)

    # Make word and frequency lists for each topic.
    sorted_topics = [
        sorted(topic.items(), key=lambda item: item[1], reverse=True)
        for topic in topics_freqs
    ]
    word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
    freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]

    # Print latex table with shaded topics.
    table_filename = "_".join(["NMF_topic_keywords", trial_params])
    table_filepath = os.path.join(results_dir, "Tables", table_filename)
    with open(table_filepath, "w") as file:
        file.write(plotting.shaded_latex_topics(freq_lists, word_lists))

    # Get topic distributions for each time slice
    num_time_slices = X.shape[0]
    topics_over_time = np.split(H, num_time_slices, axis=1)

    # Average topic distributions over time.
    avg_topics_over_time = [np.mean(topics, axis=1) for topics in topics_over_time]

    # Normalize to get the distribution over topics for each day.
    avg_topics_over_time = np.array(
        [topics / np.sum(topics) for topics in avg_topics_over_time]
    )

    # #### Visualize topic distributions
    start = datetime.datetime(2020, 2, 1, 0)
    dates = [i * datetime.timedelta(days=1) + start for i in range(num_time_slices)]
    date_strs = [date.strftime("%m-%d") for date in dates]
    y_tick_labels = [
        "{}: {}".format(", ".join(word_lists[i][0:3]), i + 1) for i in range(r)
    ]

    fig, ax = plotting.heatmap(
        avg_topics_over_time.T,
        x_tick_labels=date_strs,
        x_label="Date",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
        max_data=0.25,
    )

    # Save figure.
    fig_filename = "_".join(["NMF_tweet_representation_of_topics", trial_params])
    fig_filepath = os.path.join(
        results_dir, "Figures", folder_name, "FixedNMF", fig_filename
    )
    plotting.save_figure(fig, filepath=fig_filepath)
