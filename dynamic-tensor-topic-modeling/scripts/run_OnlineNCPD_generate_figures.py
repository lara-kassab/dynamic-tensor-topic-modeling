import datetime
import os
import pickle

import numpy as np
import seaborn as sns

from config import data_dir, results_dir
from covid19 import plotting, utils
from covid19.online_CPDL.tweets_reconstruction_OCPDL import Tweets_Reconstructor_OCPDL

# This script generates Figures 4 and 6 in the subsmission
# "On Nonnegative Matrix and Tensor Decompositions for COVID-19 TwitterDynamics"

do_batch_processing = True  # Generates Fig 4
do_sequential_processing = True  # Generates Fig 6


# Set up the OnlineNCPD Twitter tensor reconstructor

path = os.path.join(data_dir, "tweet_tensor.pickle")  # path to the tensor data
dict = pickle.load(open(path, "rb"))
X_words = dict[0]  # list of words


if do_batch_processing:

    n_topics = 20
    n_iter = 50
    sparsity = 1
    batch_size = 100
    save_file_name = (
        "top1000"
        + "_ntopics_"
        + str(n_topics)
        + "_iter_"
        + str(n_iter)
        + "_sparsity_"
        + str(sparsity).replace(".", "")
        + "_batchsize_"
        + str(batch_size)
    )

    reconstructor = Tweets_Reconstructor_OCPDL(
        path=path,
        n_components=n_topics,  # number of dictionary elements -- rank
        iterations=n_iter,  # number of iterations for the ONTF algorithm
        sub_iterations=2,
        # number of i.i.d. subsampling for each iteration of ONTF
        batch_size=1,  # number of patches used in i.i.d. subsampling
        num_patches=1,
        # number of patches that the algorithm learns from at each iteration
        segment_length=batch_size,
        alpha=sparsity,
        unfold_words_tweets=False,
    )

    W, At, Bt, H = reconstructor.train_dict(
        save_file_name=save_file_name,
        if_sample_from_tweets_mode=True,
        beta=sparsity,
        if_save=False,
    )
    CPdict = reconstructor.out(W)
    print("W.keys()", W.keys())
    print("CPdict.keys()", CPdict.keys())
    print("U0.shape", W.get("U0").shape)  # Prevalence
    print("U1.shape", W.get("U1").shape)  # Topics

    topics = W.get("U1")
    print("!!!", W.get("U1").shape)
    prevalence = W.get("U0")
    print("!!!", W.get("U0").shape)

    r = topics.shape[1]
    data_name = "top1000daily"
    folder_name = "Tweets_dictionary"

    # Condense topic representations.
    topics_freqs = []
    for i, topic in enumerate(topics.T):
        topics_freqs.append(
            {X_words[i]: topic[i] for i in reversed(topic.argsort()[-r:])}
        )
        print(
            "Topic {}: {}".format(
                i,
                ", ".join(
                    x for x in reversed(np.array(X_words)[topic.argsort()[-10:]])
                ),
            )
        )

    topics_freqs = utils.condense_topic_keywords(topics_freqs)
    num_keywords = 5

    sns.set(style="whitegrid", context="talk")

    # Make word and frequency lists for each topic.
    sorted_topics = [
        sorted(topic.items(), key=lambda item: item[1], reverse=True)
        for topic in topics_freqs
    ]
    word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
    freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]
    # Print latex table with shaded topics.
    table_filename = "NMF_{}_topic_keywords_{}_{}_topics.tex".format(
        folder_name, data_name, r
    )
    topic_latex = plotting.shaded_latex_topics(freq_lists, word_lists, min_shade=0.6)
    print("!!! topic_latex", topic_latex)

    num_time_slices = prevalence.shape[0]

    avg_topics_over_time = prevalence.T
    for i in np.arange(prevalence.shape[0]):
        avg_topics_over_time[:, i] = avg_topics_over_time[:, i] / np.sum(
            avg_topics_over_time[:, i]
        )

    # #### Visualize topic distributions
    start = datetime.datetime(2020, 2, 1, 0)
    dates = [i * datetime.timedelta(days=1) + start for i in range(num_time_slices)]
    date_strs = [date.strftime("%m-%d") for date in dates]
    # y_tick_labels = ["{}: {}".format(word_lists[i][0:2], i + 1) for i in range(r)]
    y_tick_labels = [
        str(word_lists[i][0])
        + ", "
        + str(word_lists[i][1])
        + ", "
        + str(word_lists[i][2])
        + " "
        + str(i + 1)
        for i in range(r)
    ]

    fig, ax = plotting.heatmap(
        avg_topics_over_time,
        x_tick_labels=date_strs,
        x_label="Date",
        y_tick_labels=y_tick_labels,
        y_label="Topic",
    )
    # Save figure.
    plotting.save_figure(
        fig, filepath=os.path.join(results_dir, "fig_ONCPD_heatmap.pdf")
    )


if do_sequential_processing:

    n_topics = 5
    n_iter = 100
    sparsity = 0.1
    batch_size = 30
    segment_length = batch_size
    seq_refresh_history_as = 1
    save_file_name = (
        "top1000"
        + "_ntopics_"
        + str(n_topics)
        + "_iter_"
        + str(n_iter)
        + "_sparsity_"
        + str(sparsity).replace(".", "")
        + "_batchsize_"
        + str(batch_size)
        + "_ref_history_as_"
        + str(seq_refresh_history_as)
    )

    reconstructor = Tweets_Reconstructor_OCPDL(
        path=path,
        n_components=n_topics,  # number of dictionary elements -- rank
        iterations=n_iter,  # number of iterations for the ONTF algorithm
        sub_iterations=2,
        # number of i.i.d. subsampling for each iteration of ONTF
        batch_size=1,  # number of patches used in i.i.d. subsampling
        num_patches=1,
        # number of patches that the algorithm learns from at each iteration
        segment_length=batch_size,
        alpha=sparsity,
        unfold_words_tweets=False,
    )

    seq_dict = reconstructor.train_sequential_dict(
        save_file_name=save_file_name,
        slide_window_by=segment_length,
        refresh_history_as=seq_refresh_history_as,
        beta=0.7,
    )
    print("seq_dict.keys()", seq_dict.keys())

    M = len(seq_dict.keys())
    W0 = seq_dict.get("W0")
    W1 = seq_dict.get("W1")
    W2 = seq_dict.get("W2")
    seq_dict = [W0.get("U1"), W1.get("U1"), W2.get("U1")]
    print("!!!", W0.get("U1").shape)
    prevalence = [W0.get("U0"), W1.get("U0"), W2.get("U0")]
    print("!!!", W0.get("U0").shape)

    fig = plotting.display_dictionaries(
        seq_dict=seq_dict,
        X_words=X_words,
        topic_freqs=prevalence,
        num_samples=100,
        num_words_from_topic=10,
        xlabels=["Topic {}".format(i + 1) for i in range(W0.get("U0").shape[1])],
        ylabels=["February", "March", "April"],
        show_plot=False,
    )
    plotting.save_figure(fig, filepath=os.path.join(results_dir, "fig_OCPDL.pdf"))
