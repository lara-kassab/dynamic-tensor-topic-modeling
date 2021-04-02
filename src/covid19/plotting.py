import datetime

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter
from wordcloud import WordCloud

sns.set(style="whitegrid", font_scale=1.5, context="talk")

"""
For details on the params below, see the matplotlib docs:
https://matplotlib.org/users/customizing.html
"""

plt.rcParams["axes.edgecolor"] = "0.6"
plt.rcParams["axes.labelsize"] = 26
plt.rcParams["figure.dpi"] = 200
# plt.rcParams["font.family"] = "serif"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["legend.columnspacing"] *= 0.8
plt.rcParams["legend.edgecolor"] = "0.6"
plt.rcParams["legend.markerscale"] = 1.0
plt.rcParams["legend.framealpha"] = "1"
plt.rcParams["legend.handlelength"] *= 1.5
plt.rcParams["legend.numpoints"] = 2
plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.major.pad"] = -3
plt.rcParams["ytick.major.pad"] = -2
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["figure.figsize"] = [12.0, 6.0]


def save_figure(fig, filepath=None, bbox_inches="tight", pad_inches=0.1):
    """Save figure in filepath."""
    if filepath is None:
        raise Exception("Filepath must be specified if save_fig=True.")
    fig_kwargs = {"bbox_inches": "tight", "pad_inches": 0.1, "transparent": True}
    fig.savefig(filepath + ".png", **fig_kwargs)
    fig.savefig(filepath + ".pdf", **fig_kwargs)
    # plt.close()


def plot_data(
    data, x=None, plot_type="lineplot", figsize=[12.0, 6.0],
):
    """
    Args:
        data (2d array): 2d array with dimensions: num_topics x num_time_slices

    Examples:

        A simple example.

        >>> import numpy as np
        >>> data = np.arange(40).reshape([4,10])
        >>> fig, ax = plot_data(data)
        >>> fig, ax = plot_data(data, plot_type='lineplot')
        >>> fig, ax = plot_data(data, plot_type='stackplot')

        An example using a Pipeline.

        >>> import numpy as np
        >>> from functools import partial
        >>> from covid19.utils import Pipeline
        >>> data = [i*np.arange(10).T for i in range(1, 20)]
        >>> data_pipeline = Pipeline(data)
        >>> data_pipeline = data_pipeline.add_map(partial(np.expand_dims, axis=1))
        >>> topic_distributions = np.concatenate(list(data_pipeline), axis=1)
        >>> fig, ax = plot_data(topic_distributions, plot_type='stackplot')
        >>> fig, ax = plot_data(topic_distributions, plot_type='lineplot')
    """
    # Get dimensions.
    num_topics, num_time_slices = data.shape
    sns.set_palette(sns.husl_palette(num_topics))

    # Create labels.
    # TODO: pass labels in as argument.
    labels = ["Topic {}".format(i) for i in range(1, num_topics + 1)]

    if x is None:
        x = np.arange(num_time_slices)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data.
    if plot_type == "lineplot":
        for topic in range(num_topics):
            plt.plot(x, data[topic, :], label=labels[topic])
    if plot_type == "stackplot":
        plt.stackplot(x, data, labels=labels)

    # Put the legend out of the figure.
    plt.legend(
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        borderaxespad=0.0,
        prop={"size": 10},
    )

    plt.xticks(rotation=45)
    if isinstance(x[0], datetime.datetime):
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)

    return fig, ax


def plot_keyword_bars_from_dictionary(
    words_by_topics,
    words,
    num_keywords=5,
    normalize_freq_sums=True,
    figsize=[12.0, 6.0],
):
    """
    Args:
        words_by_topics : 2darray(float)
            Array of word frequencies for each topic. Each row is a distinct word, the columns are different topics.
        words : 1darray(str)
            Words corresponding to the rows of ``words_by_topics``.
        num_keywords: int
            The number of keywords to include for each topic.
        normalize_freq_sums: bool
            Whether to normalize the word topic associations so that the bars all have the same overall length.


    Examples:
        >>> num_words = 10
        >>> num_topics = 5
        >>> data = np.random.rand(num_words * num_topics).reshape((num_words, num_topics))
        >>> words = np.array(["word {}".format(i) for i in range(num_words)])
        >>> fig, ax = plot_keyword_bars_from_dictionary(data, words, num_keywords = 3, normalize_freq_sums=True)
    """
    # words_by_topics = words_by_topics / words_by_topics.sum(axis=0, keepdims=True)

    n_words, n_topics = words_by_topics.shape

    if num_keywords > n_words:
        raise Exception("You are asking for more words than exist. Try less words.")

    # TODO: Wrap in a function?
    ordered_word_idxs = np.argsort(words_by_topics, axis=0)
    top_n_word_idxs = np.flipud(ordered_word_idxs[-num_keywords:, :])
    ordered_word_freqs = np.sort(words_by_topics, axis=0)
    top_n_word_freqs = np.flipud(ordered_word_freqs[-num_keywords:, :])

    topic_words = [words[word_ids] for word_ids in top_n_word_idxs.T]

    return plot_keyword_bars(
        top_n_word_freqs.T,
        topic_words,
        normalize_freq_sums=normalize_freq_sums,
        figsize=figsize,
    )


def plot_keyword_bars(
    ordered_word_freqs, words, normalize_freq_sums=True, figsize=[12.0, 6.0],
):
    """
    Args:
        words_by_topics : 2darray(float)
            Array of word frequencies for each topic.
            Each rows corresponds to a topic.
            Each row contains weights for the bar widths.
        words : 2darray(str)
            Words corresponding to topics in the rows of ``words_by_topics``.
        normalize_freq_sums: bool
            Whether to normalize the word topic associations so that the bars all have the same overall length.


    Examples:
        >>> num_words = 5
        >>> num_topics = 10
        >>> data = np.random.rand(num_words * num_topics).reshape((num_words, num_topics))
        >>> words = [["word {}".format(i) for i in range(num_words)] * num_topics]
        >>> fig, ax = plot_keyword_bars(data, words, normalize_freq_sums=True)
    """
    ordered_word_freqs = np.array(ordered_word_freqs)
    n_topics, n_words = ordered_word_freqs.shape

    # Setup the figure.
    fig, ax = plt.subplots(figsize=figsize)
    ax.invert_yaxis()

    # Normalize rows.
    if normalize_freq_sums:
        ordered_word_freqs /= ordered_word_freqs.sum(axis=1)[:, None]
        ax.xaxis.set_visible(False)

    # TODO: Consider log scaled axes to make smaller bars more readable.
    # This is only really necessary if you are not normalizing, but who cares.
    ax.set_xlim(0, ordered_word_freqs.sum(axis=1).max())

    # TODO: How about using a color gradient?
    colors = plt.get_cmap("Blues")(np.linspace(0.85, 0.25, n_words))

    labels = ["Topic {}".format(i) for i in range(1, n_topics + 1)]

    bar_starts = np.zeros(n_topics)
    for word_i in range(n_words):
        bar_widths = ordered_word_freqs[:, word_i]
        bar_centers = bar_starts + bar_widths / 2

        ax.barh(
            labels,
            bar_widths,
            left=bar_starts,
            height=0.95,
            label=str(word_i),
            color=colors[word_i],
        )

        # TODO: Wrap in a function
        # If the color of the bg is light, use 'darkgrey', else use 'white'
        r, g, b, _ = colors[word_i]
        text_color = "white" if r * g * b < 0.5 else "dimgrey"

        ngrams = [word_list[word_i] for word_list in words]
        for y, (x, ngram) in enumerate(zip(bar_centers, ngrams)):
            ax.text(
                x,
                y,
                ngram,  # .replace(" ", "\n"),
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )

        bar_starts += bar_widths

    return fig, ax


def heatmap(
    data,
    x_tick_labels=None,
    x_label="",
    y_tick_labels=None,
    y_label="",
    figsize=(7, 9),
    max_data=None,
):
    """Plot heatmap.

    Args:
        data: (2d array) data to be plotted (topics x date)
        x_tick_labels (list of str)

    Returns:
        fig
        ax
    """

    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        data,
        rasterized=True,
        vmax=max_data,
        cbar_kws=dict(use_gridspec=False, location="top"),
    )

    plt.xticks(rotation=45)
    plt.yticks(np.arange(0, data.shape[0], 1.0) + 0.5, rotation=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if y_tick_labels is None:
        y_tick_labels = [topic_num + 1 for topic_num in range(data.shape[0])]
    ax.set_yticklabels(y_tick_labels)

    if x_tick_labels is not None:
        labels = [x_tick_labels[int(item.get_text())] for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

    return fig, ax


def shaded_latex_topics(freq_lists, word_lists, min_shade=0.6):
    """Generate latex table with keywords shaded by frequency value.
    Based on stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights

    Args:
        freq_lists (list of lists of floats):
        word_lists (list of lists of strings):
        min_shade (float): min shade to be printed so that values are visible.

    Returns:
    (string) Latex table when printed.
    """

    if len(freq_lists) != len(word_lists):
        raise Exception("Frequency and word lists must have the same length.")

    if len(word_lists[0]) == 0:
        raise Exception("Word list is empty.")

    cmap = matplotlib.cm.Greys
    colored_string = "\\begin{{tabular}}{{|l{}|}} \\hline \n".format(
        "".join(["c" for i in range(len(word_lists[0]))])
    )

    for i, (freqs, words) in enumerate(zip(freq_lists, word_lists)):
        colored_string += "Topic {}: &".format(i + 1)

        # Normalize topic frequencies.
        freqs = np.array(freqs) / sum(freqs)

        # Get list of color shades.
        colors = [
            matplotlib.colors.rgb2hex(cmap(shifted_freqs))[1:]
            for shifted_freqs in (1 - min_shade) * freqs + min_shade
        ]

        # Add topic keywords to table.
        colored_string += "&".join(
            [
                "\\textcolor[HTML]{"
                + color
                + "}{"
                + "{} ({:.2f})".format(word, freq)
                + "} "
                for freq, word, color in zip(freqs, words, colors)
            ]
        )
        colored_string += "\\\\ \\hline \n"

    colored_string += "\\end{tabular}"
    return colored_string


def shaded_html_topics(freq_lists, word_lists):
    """Based on stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights"""

    if len(freq_lists) != len(word_lists):
        raise Exception("Frequency and word lists must have the same length.")

    cmap = matplotlib.cm.Greys
    template = '<span class="barcode"; style="color: {}">{}</span>'
    black = matplotlib.colors.rgb2hex(cmap(1.0)[:3])
    colored_string = ""

    for i, (freqs, words) in enumerate(zip(freq_lists, word_lists)):
        # Normalize topic frequencies.
        sum_of_values = sum(freqs)

        colored_string += "<p>" + template.format(
            black, "&nbsp Topic&nbsp{}:&nbsp".format(i + 1)
        )

        for weight, word in zip(freqs, words):
            # Normalize weights.
            weight /= sum_of_values
            # Make smallest weights visible.
            min_shade = 0.4
            shifted_weight = (1 - min_shade) * weight + min_shade
            color = matplotlib.colors.rgb2hex(cmap(shifted_weight)[:3])
            colored_string += template.format(
                color, "&nbsp" + word + "&nbsp" + "({:.2f})".format(weight) + "&nbsp"
            )
        colored_string += "</p>"
    return colored_string


def grey_color_func(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)


def random_color_func(
    word=None,
    font_size=None,
    position=None,
    orientation=None,
    font_path=None,
    random_state=None,
):
    hue = int(360.0 * 21.0 / 255.0)
    sat = int(100.0 * 255.0 / 255.0)
    light = int(100.0 * float(random_state.randint(180, 255)) / 255.0)

    return "hsl({}, {}%, {}%)".format(hue, sat, light)


def display_dictionary_CP(
    W,
    X_words,
    save_fig_name=None,
    num_word_sampling_from_topics=100,
    num_top_words_from_each_topic=10,
    if_plot=False,
):
    # X_words = list of words in the original tweets tensor (first component of the tensor)
    # W = dictionary({'U0': time modes}, {'U1': topic modes}, {'U2': tweets mode})

    U0 = W.get("U0")  # dict for time mode
    U1 = W.get("U1")  # dict for words mode (topics)

    n_components = U0.shape[1]
    num_rows = np.floor(np.sqrt(n_components)).astype(int)
    num_cols = np.ceil(np.sqrt(n_components)).astype(int)

    # topic mode
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(9, 9),
        subplot_kw={"xticks": [], "yticks": []},
    )
    for ax, i in zip(axs.flat, range(n_components)):
        print("Sampling from the %i th topic" % i)
        patch = U1[:, i]
        idxs = np.argsort(patch)
        idxs = np.flip(idxs)
        # print('patch[idxs]', patch[idxs])
        idxs = idxs[0:num_top_words_from_each_topic]
        # print('idxs', idxs)
        patch_reduced = patch[idxs]
        # probability distribution on the words given by the ith topic vector
        dist = patch_reduced.copy() / np.sum(patch_reduced)

        # Randomly sample a word from the corpus according to the PMF "dist" multiple times
        # to generate text data corresponding to the ith topic, and then generate its wordcloud
        list_words = []
        a = np.arange(len(patch_reduced))

        for j in range(num_word_sampling_from_topics):
            rand_idx = np.random.choice(a, p=dist)
            list_words.append(X_words[idxs[rand_idx]])
            # print('X_words[idx]', X_words[idx])
        # print(list_words)
        Y = " ".join(list_words)
        wordcloud = WordCloud(
            background_color="black", relative_scaling=1, width=400, height=400
        ).generate(Y)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle("Topics mode", fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
    plt.savefig("Tweets_dictionary/fig_topics" + "_" + str(save_fig_name))
    if if_plot:
        plt.show()

    # time mode
    fig, axs = plt.subplots(
        nrows=1, ncols=1, figsize=(6, 6), subplot_kw={"xticks": [], "yticks": []}
    )

    # Normalize code matrix
    for i in np.arange(n_components):
        U0[:, i] /= np.sum(U0[:, i])
        print("np.sum(U0[:,i])", np.sum(U0[:, i]))

    axs.imshow(U0, cmap="viridis", interpolation="nearest", aspect="auto")
    print("time_mode_shape")
    print("time_mode:", U0[:, i].reshape(1, -1))

    plt.tight_layout()
    plt.suptitle("Time mode", fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.00, 0.00)
    plt.savefig("Tweets_dictionary/fig_temporal_modes" + "_" + str(save_fig_name))
    if if_plot:
        plt.show()


def display_dictionaries(
    seq_dict,
    X_words,
    topic_freqs=None,
    save_fig_name=None,
    num_samples=100,
    num_words_from_topic=10,
    show_plot=False,
    xlabels=None,
    ylabels=None,
    figsize=(10, 6),
):
    """Plot wordcloud dictionary representations.

    Args:
        seq_dict (list of words x topics ndarrays):
            list of matrices whose columns correspond to topics
        X_words (list of strings): ordered list of words in the vocabulary
        topic_freqs (list of days x topics ndarrays): the prevalence of each
            topic for each dictionary matrix

    Returns:
        fig
    """
    nrows = len(seq_dict)  # Number of dictionaries.
    ncols = seq_dict[0].shape[1]  # Number of topics.

    # Check that inputs are valid.
    if xlabels is not None and len(xlabels) != ncols:
        raise Exception(
            "xlabels has length {}, expected {}".format(len(xlabels), ncols)
        )
    if ylabels is not None and len(ylabels) != nrows:
        raise Exception(
            "ylabels has length {}, expected {}".format(len(ylabels), nrows)
        )
    if topic_freqs is not None and len(topic_freqs) != nrows:
        raise Exception(
            "topic_freqs has length {}, expected {}".format(len(topic_freqs), nrows)
        )

    fig = plt.figure(figsize=figsize, constrained_layout=False)

    # Make outer gridspec.
    outer_grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.05, hspace=0.05)
    # Make nested gridspecs.
    for row, W in zip(range(nrows), seq_dict):
        for col in range(ncols):
            inner_grid = outer_grid[row * ncols + col].subgridspec(
                1, 12, wspace=0.00, hspace=0.05
            )
            ax = fig.add_subplot(inner_grid[:, :-2])

            # Add plot labels and remove remainder of axis.
            if col == 0:
                ax.set_ylabel(ylabels[row])
            if row == nrows - 1:
                ax.set_xlabel(xlabels[col])
            ax.set_frame_on(False)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])

            # Sample from the 'col'th topic.
            patch = W[:, col]
            patch_idxs = np.flip(np.argsort(patch))
            patch_idxs = patch_idxs[0:num_words_from_topic]
            patch_reduced = patch[patch_idxs]
            # Probability distribution over words given by the col'th topic.
            dist = patch_reduced.copy() / np.sum(patch_reduced)

            # Randomly sample words from the corpus according to the PMF "dist" multiple times
            # to generate text data corresponding to the ith topic, and then generate its wordcloud.
            idxs = [
                np.random.choice(num_words_from_topic, p=dist)
                for i in range(num_samples)
            ]
            list_words = [X_words[patch_idxs[idx]] for idx in idxs]

            Y = " ".join(list_words)
            wordcloud = WordCloud(
                background_color="black",
                relative_scaling=0,
                prefer_horizontal=1,
                width=400,
                height=400,
            ).generate(Y)
            ax.imshow(
                wordcloud.recolor(color_func=random_color_func, random_state=3),
                interpolation="bilinear",
                aspect="auto",
            )

            # Add heatmap bar of topic prevalence over time.
            if topic_freqs is not None:
                ax = fig.add_subplot(inner_grid[:, -2:])
                ax.imshow(
                    topic_freqs[row][:, col].reshape(1, -1).T,
                    aspect="auto",
                    interpolation="nearest",
                )
                ax.axis("off")

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig
