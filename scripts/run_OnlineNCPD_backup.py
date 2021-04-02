import datetime
import pickle

import numpy as np
import seaborn as sns

# from config import results_dir
from covid19 import plotting, utils
from covid19.online_CPDL import tweets_reconstruction_OCPDL

path_data = "../Data/data_tensor_top1000.pickle"
dict = pickle.load(open(path_data, "rb"))
X_words = dict[0]  # list of words

is_sequential = False

if is_sequential:
    path = "../Data/sequential_dict_learned_CPDL_top1000_ntopics_5_iter_100_sparsity_01_batchsize_30_ref_history_as_1.npy"
    factors_dict = np.load(path, allow_pickle=True).item()
    M = len(factors_dict.keys())
    W0 = factors_dict.get("W0")
    W1 = factors_dict.get("W1")
    W2 = factors_dict.get("W2")
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
    # fig_filename = "_".join(["ONMF_dictionaries", trial_params])
    # fig_filepath = os.path.join(results_dir, "Figures", folder_name, "ONMF", fig_filename)
    plotting.save_figure(fig, filepath="../Data/fig_OCPDL.pdf")

else:
    path = "../Data/dict_learned_CPDL_top1000_ntopics_20_iter_50_sparsity_1_batchsize_100_ref_history_as_1.npy"
    W = np.load(path, allow_pickle=True).item()
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
    # table_filepath = os.path.join(overleaf_dir, "Tables", table_filename)
    # with open(table_filepath, "w") as file:
    #   file.write(plotting.shaded_latex_topics(freq_lists, word_lists))
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
    fig_filename = "ONCPD_tweet_representation_of_topics_{}_{}_topics".format(
        data_name, r
    )
    fig_filepath = "../Data/" + fig_filename
    plotting.save_figure(fig, filepath="../Data/fig_ONCPD_heatmap.pdf")
