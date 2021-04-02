"""Utility functions for applying nmf."""


def topic_keywords(num_keywords, topic_dict, idx_to_word, verbose=True):
    """Print the most representative words for each topic.

    Args:
        num_keywords: The number of keywords to include for each topic.
        topic_dict: The dictionary W mapping words to topics.
        idx_to_word: Mapping of indices to words for the bag of words
            representation.
        verbose: Whether to print keywords for each topics.

    Returns:
        List of lists with topic keywords for each topic.
    """
    topic_keyword_list = []
    for i in range(topic_dict.shape[1]):
        topic = topic_dict[:, i]
        topic_keyword_list.append(
            [x for x in reversed(idx_to_word[topic.argsort()[-num_keywords:]])]
        )

        # Print the most representative words for each topic.
        if verbose:
            print("Topic {}: {}\n".format(i + 1, topic_keyword_list[-1]))
    return topic_keyword_list
