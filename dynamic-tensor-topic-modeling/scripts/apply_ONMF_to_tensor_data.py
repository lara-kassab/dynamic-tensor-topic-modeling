import datetime
import os
import pickle
from functools import partial
from itertools import product

import numpy as np

from config import data_dir, results_dir
from covid19 import plotting, utils
from covid19.online_nmf import onmf
from covid19.utils import Pipeline

ranks = [5, 20]
data_names = ["top_1000_daily"]
folder_names = ["top"]
betas = [0.7]
# If the regularization is too large, the codes will only contain 0.
regularization_params = [0]
parameter_combos = product(
    ranks, zip(data_names, folder_names), betas, regularization_params
)

# Execute experiments for parameter combinations.
for r, (data_name, folder_name), beta, regularization in parameter_combos:
    np.random.seed(100)
    trial_params = "{}_{}_topics_beta_{}_reg_{}".format(
        data_name, r, beta, regularization
    )
    print(trial_params)
    # Load data.
    if data_name == "random_1000_daily":
        directory = os.path.join(
            data_dir,
            "data_tensor1000_random_tweets_daily_from_2020-02-01-00_to_2020-05-01-00.pickle",
        )  # Fill
    if data_name == "top_1000_daily":
        directory = os.path.join(data_dir, "tweet_tensor.pickle")  # Fill
    with open(directory, "rb") as file:
        [features, X] = pickle.load(file)

    # Apply ONMF.
    nmf = onmf.Online_NMF(
        X,
        num_topics=r,
        regularization_param=regularization,
        iterations=100,
        batch_size=50,
        beta=beta,
    )
    W, A, B, seq_dict = nmf.train_dict(get_dicts=[30, 60, 90])

    # Get the most representative words for each topic.
    topics_freqs = []
    for i, topic in enumerate(W.T):
        topics_freqs.append(
            {features[i]: topic[i] for i in reversed(topic.argsort()[-20:])}
        )
    topics_freqs = utils.condense_topic_keywords(topics_freqs)

    # Make word and frequency lists for each topic.
    num_keywords = 5
    sorted_topics = [
        sorted(topic.items(), key=lambda item: item[1], reverse=True)
        for topic in topics_freqs
    ]
    word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
    freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]

    # Write latex table with shaded topics.
    table_filename = "_".join(["ONMF_keywords", trial_params])
    table_filepath = os.path.join(results_dir, "Tables", table_filename)
    with open(table_filepath, "w") as file:
        file.write(plotting.shaded_latex_topics(freq_lists, word_lists))

    # Get code matrices for each time slice. The resulting code matrices will have
    topic_distr_pipeline = Pipeline(X)
    topic_distr_pipeline.add_map(partial(nmf.get_code_matrix, W=W))

    # Take the mean of the topic distribution for each file.
    topic_distr_pipeline.add_map(partial(np.mean, axis=1))

    # Normalize each distribution.
    topic_distr_pipeline.add_map(utils.to_proportion)

    # Create array with dim: data_dim x num_samples.
    topic_distr_pipeline.add_map(partial(np.expand_dims, axis=1))
    topic_distributions = np.concatenate(list(topic_distr_pipeline), axis=1)

    # Plot figure.
    # Get dates of time slices.
    start = datetime.datetime(2020, 2, 1, 0)
    dates = [i * datetime.timedelta(days=1) + start for i in range(X.shape[0])]
    date_strs = [date.strftime("%m-%d") for date in dates]
    y_tick_labels = [
        "{}: {}".format(", ".join(word_lists[i][0:3]), i + 1) for i in range(r)
    ]
    fig, ax = plotting.heatmap(
        topic_distributions,
        x_tick_labels=date_strs,
        x_label="Date",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
    )

    # Save figure.
    fig_filename = "_".join(["ONMF_tweet_representation_of_topics", trial_params])
    fig_filepath = os.path.join(
        results_dir, "Figures", folder_name, "ONMF", fig_filename
    )
    plotting.save_figure(fig, filepath=fig_filepath)

    # Get average topic frequency for topics from each month.
    topic_freqs = []
    for months_data in [X[0:29], X[29:60], X[60:]]:
        topic_distr_pipeline = Pipeline(months_data)
        topic_distr_pipeline.add_map(partial(nmf.get_code_matrix, W=W))

        # Take the mean of the topic distribution for each file.
        topic_distr_pipeline.add_map(partial(np.mean, axis=1))

        # Normalize each distribution.
        topic_distr_pipeline.add_map(utils.to_proportion)

        # Create array with dim: data_dim x num_samples.
        topic_distr_pipeline.add_map(partial(np.expand_dims, axis=1))
        topic_freqs.append(np.concatenate(list(topic_distr_pipeline), axis=1).T)

    # Plot and save dictionary visualizations with topic frequencies per month..
    fig = plotting.display_dictionaries(
        seq_dict,
        features,
        topic_freqs=topic_freqs,
        num_samples=100,
        num_words_from_topic=10,
        xlabels=["Topic {}".format(i + 1) for i in range(W.shape[1])],
        ylabels=["February", "March", "April"],
    )
    fig_filename = "_".join(["ONMF_dictionaries_w_freqs", trial_params])
    fig_filepath = os.path.join(
        results_dir, "Figures", folder_name, "ONMF", fig_filename
    )
    plotting.save_figure(fig, filepath=fig_filepath)
