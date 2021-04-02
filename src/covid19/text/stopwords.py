"""Functions related to the management of stopwords."""

import string
from itertools import filterfalse

from ._nltk_stopwords import nltk_stopwords

# Augment stopwords list.
_number_strs = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]
_other_unwanted_things = [
    "’",
    "”",
    "“",
    "amp",
    "https",
    "covid19ph",
    "covid",
    "corona virus",
    "coronavirusupdates",
    "coronavirusuk",
    "co",
    "coronaviruslockdown",
    "covid19pandemic",
    "coronavid19",
    "indiafightscorona",
    "covidiot",
    "coronaalert",
    "coronavirustruth",
    "coronavirususa",
    "covidiots",
    "coronaviruses",
    "2019ncov",
    "2019 ncov",
    "coronaviruswuhan",
    "coronavirusinindia",
    "corona",
    "covidー19",
    "19",
    "sarscov2",
    "cov",
    "coronavirius",
    "coronaviruspandemic",
    "coronavirusupdate",
    "nomeat_nocoronavirus",
    "virus",
    "covid19",
    "covid2019",
    "covid19uk",
    "ncov2019",
    "coronaviruschina",
    "chinacoronavirus",
    "coronavirus",
    "coronavirusoutbreak",
    "coronavirus co",
    "ncov",
    "coronaviruschallenge",
    "covid 19",
    "coronavirusindia",
    "covid_19",
    "wuhancoronavirus",
    "coronavirusus",
    "coronapocalypse",
    "covid19india",
    "coronalockdown",
    "coronaoutbreak",
    "2019",
]

_stopword_sublists = [
    nltk_stopwords,
    _number_strs,
    string.punctuation,
    _other_unwanted_things,
]
# Combine sublists into single list.
stopwords = list(set().union(*_stopword_sublists))


def is_stopword(word):
    """Check if a word is considered a stopword.

    Args:
        word (str): The word under consideration.

    Returns:
        bool: True if the word is a stopword, False otherwise.

    Examples:
        >>> is_stopword('the')
        True

        >>> is_stopword('apple')
        False
    """
    return word in stopwords


def remove_stopword_tokens(tokens):
    """Remove stopwords from tokens.

    Args:
        tokens (iterable): A token iterator

    Returns:
        iterable: A token iterator which skips the stop words

    Examples:
        >>> tokens = ['the', 'a', 'apple', '.', ',', 'coronavirus']
        >>> tokens_gen = remove_stopword_tokens(tokens)
        >>> list(tokens_gen)
        ['apple']
    """
    return filterfalse(is_stopword, tokens)
