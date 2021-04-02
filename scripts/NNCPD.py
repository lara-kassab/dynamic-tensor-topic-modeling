import datetime
import os
import pickle
import time

import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

from covid19 import plotting, utils


def NNCPD(
    data_name,
    local_path,
    tensor_filename,
    rank,
    num_time_slices=90,
    num_keywords=5,
    compute_NNCPD=1,
    factors_path=None,
    weights_path=None,
):
    """
    Compute the NNCPD of a TFIDF tensor data (dimensions: timeframes-by-features-by-documents).
    Extract keywords representation of topics and visualize all factor matrices.

    Args:
        data_name (str): descriptive string of the data used and the chosen rank
        local_path (str): directory containing pickle file of the data, and where the data files will be saved
        tensor_filename (str): name of the pickle file containing the tensor and features data.
        rank (int): input rank of NNCPD
        num_keywords (int): number of keywords to extract for the representation of topics
        compute_NNCPD (boolean): 0 to compute NNCPD and save factors, 1 to load saved NNCPD factors
        factors_path (str): path to file in "local_path" to save the NNCPD factors
        weights_path (str): path to filein "local_path" to save the NNCPD weights
    """

    # Check if NNCPD weights are produced (depend on TensorLy version)
    flag = 0

    # File names
    if factors_path is None:
        factors_path = data_name + "_factors.pickle"

    if weights_path is None:
        weights_path = data_name + "_weights.pickle"

    # Load data tensor
    directory = os.path.join(local_path, tensor_filename)
    [features, X] = pickle.load(open(directory, "rb"))
    print(f"The data tensor has shape {X.shape}.")

    # Perform NNCPD to obtain the factor matrices
    if compute_NNCPD == 1:
        print(f"Running NNCPD with rank {rank}...")
        start_time = time.time()
        factors = non_negative_parafac(X, rank=rank)
        end_time = time.time() - start_time
        print("Done!")
        print(f"It took {end_time/60 :.3f} minutes to run.")

        if len(factors) == 2:
            print("NNCPD produced weights.")
            flag = 1
            weights, factors = factors

        pathname_save = os.path.join(local_path, factors_path)
        with open(pathname_save, "wb") as f:
            pickle.dump(factors, f)
        print("Factors are saved.")

        if flag == 1:
            pathname_save = os.path.join(local_path, weights_path)
            with open(pathname_save, "wb") as f:
                pickle.dump(weights, f)
            print("Weights are saved.")
            if weights.all() == 1:
                print("The factors are not normalized.")
            else:
                print("The factors are normalized.")

    # Load saved NNCPD factors
    if compute_NNCPD == 0:
        pathname_load = os.path.join(local_path, factors_path)
        factors = pickle.load(open(pathname_load, "rb"))
        print("The NNCPD factors are loaded.")

        pathname_load = os.path.join(local_path, weights_path)
        weights = pickle.load(open(pathname_load, "rb"))
        print("The NNCPD weights are loaded.")

    rel_error = tl.norm(X - tl.kruskal_to_tensor([weights, factors]), 2) / tl.norm(X, 2)
    print(f"The relative error is {rel_error}.")

    # Obtain the shape of the factor matrices
    print(f"There are {len(factors)} NNCPD factors and have shape:")
    for i in range(len(factors)):
        print(factors[i].shape)

    # Condense topic representations.
    topics_freqs = []
    for i, topic in enumerate(factors[1].T):
        topics_freqs.append(
            {features[i]: topic[i] for i in reversed(topic.argsort()[-10:])}
        )
        print(
            "Topic {}: {}".format(
                i,
                ", ".join(
                    x for x in reversed(np.array(features)[topic.argsort()[-10:]])
                ),
            )
        )

    topics_freqs = utils.condense_topic_keywords(topics_freqs)

    # Define word and frequency lists for each topic.
    sorted_topics = [
        sorted(topic.items(), key=lambda item: item[1], reverse=True)
        for topic in topics_freqs
    ]
    word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
    freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]
    print(plotting.shaded_latex_topics(freq_lists, word_lists))

    # Normalized time-topic matrix
    time_norm = factors[0] / factors[0].sum(axis=1)[:, None]

    # Dates Labels
    start = datetime.datetime(2020, 2, 1, 0)
    dates = [i * datetime.timedelta(days=1) + start for i in range(num_time_slices)]
    date_strs = [date.strftime("%m-%d") for date in dates]
    y_tick_labels = [
        "{}: {}".format(", ".join(word_lists[i][0:3]), i + 1) for i in range(rank)
    ]

    # labels
    word_strs = [i + 1 for i in range(factors[1].shape[0])]
    tweet_strs = [i + 1 for i in range(factors[2].shape[0])]

    # Produce various plots for the factor matrices of NNCPD
    # Topic-Time matrix with labels
    fig, ax = plotting.heatmap(
        factors[0].T,
        x_tick_labels=date_strs,
        x_label="Date",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
    )
    fig_filename = "Topic_Time_{}".format(data_name)
    fig_filepath = os.path.join(local_path, fig_filename)
    plotting.save_figure(fig, filepath=fig_filepath)

    # Topic-Time matrix normalized with no labels
    fig, ax = plotting.heatmap(
        time_norm.T, x_tick_labels=date_strs, x_label="Date", y_label="Topic",
    )
    fig_filename = "Topic_Time_Normalized_{}".format(data_name)
    fig_filepath = os.path.join(local_path, fig_filename)
    plotting.save_figure(fig, filepath=fig_filepath)

    # Topic-Time matrix normalized with labels
    fig, ax = plotting.heatmap(
        time_norm.T,
        x_tick_labels=date_strs,
        x_label="Date",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
    )
    fig_filename = "Topic_Time_Normalized_labels_{}".format(data_name)
    fig_filepath = os.path.join(local_path, fig_filename)
    plotting.save_figure(fig, filepath=fig_filepath)

    # Topic-Word matrix with labels
    fig, ax = plotting.heatmap(
        factors[1].T,
        x_tick_labels=word_strs,
        x_label="Word",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
    )
    fig_filename = "Topic_Word_{}".format(data_name)
    fig_filepath = os.path.join(local_path, fig_filename)
    plotting.save_figure(fig, filepath=fig_filepath)

    # Topic-Tweet matrix with no labels
    fig, ax = plotting.heatmap(
        factors[2].T, x_tick_labels=tweet_strs, x_label="Tweets", y_label="Topic",
    )
    fig_filename = "Topic_Tweet_{}".format(data_name)
    fig_filepath = os.path.join(local_path, fig_filename)
    plotting.save_figure(fig, filepath=fig_filepath)
