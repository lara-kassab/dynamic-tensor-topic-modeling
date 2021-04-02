from nltk.util import everygrams


def get_ngrams(max_len, tokens, delimiter=" "):
    """Get ngrams (sequences of consecutive tokens) from tokens.

    ngrams are sequences of consecutive tokens. Return an iterator of all
    ngrams of length at most max_len represented as the concatenation of the
    constituent tokens, delimited by delimiter.

    Args:
        max_len (int): Max length of ngram to consider.
        tokens (iterable of str): Token iterator.
        delimiter (str, optional): Separator to use between tokens in an ngram.

    Returns:
        iterable of str: String representations of each ngram.

    Examples:
        To use `get_ngrams` directly on an iterable of tokens:

        >>> list(get_ngrams(2, ["a", "b", "c"]))
        ['a', 'b', 'c', 'a b', 'b c']

        To use `get_ngrams` on a stream of token iterables:

        >>> tokens_gen = iter([["a", "b", "c"],
        ...                    ["d", "e", "f"]])
        >>> from functools import partial
        >>> ngrams_gen = map(partial(get_ngrams, 2), tokens_gen)
        >>> from covid19.utils import listify_nested_iterables
        >>> listify_nested_iterables(ngrams_gen)
        [['a', 'b', 'c', 'a b', 'b c'], ['d', 'e', 'f', 'd e', 'e f']]
    """
    # Gotcha: everygrams doesn't work with iterables. Only lists.
    ngrams = everygrams(list(tokens), max_len=max_len)
    return map(delimiter.join, ngrams)
