"""Utility functions not related to any particular module."""
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import cached_property, wraps

import numpy as np


class Pipeline:
    """Apply a sequence of manipulations to a stream of items.

    Args:
        stream (iterable): A stream of things to be filtered and transformed.
        steps (list of (applier, to_apply)): Manipulations to apply to the
            stream. Each manipulation will be applied by
            `applier(to_apply, stream)`.
        precompute_len (bool): Whether to compute the length at initialization.
            Recommended when no filters are applied to the stream and the
            pipeline involves expensive operations.

    Attributes:
        steps (list of (applier, to_apply)): List of manipulations to apply.

    Returns:
        iterable: A stream derived from the input stream by applying the
            specified manipulations.

    Example:
        Given the input stream

        >>> stream = range(10)

        The following pipeline will remove odd numbers and quadruple the
        remaining numbers from the stream using the following helpers.

        >>> is_even = lambda x: x%2==0
        >>> quadruple = lambda x: x*4

        The steps in the pipeline are provided as a list of manipulations to
        apply, in the order in which they should be applied.

        >>> steps = [(filter, is_even),
        ...          (map, quadruple)]
        >>> pipeline = Pipeline(stream, steps)
        >>> len(pipeline)
        5
        >>> list(pipeline)
        [0, 8, 16, 24, 32]

        To instead remove the even numbers before quadrupling, you can make use
        of the `negate` function.

        >>> stream = range(10)
        >>> steps = [(filter, negate(is_even)),
        ...          (map, quadruple)]
        >>> pipeline = Pipeline(stream, steps)
        >>> list(pipeline)
        [4, 12, 20, 28, 36]

        To add additional steps to the pipeline, use `pipeline.add_filter`
        or `pipeline.add_map`.

        >>> stream = range(10)
        >>> pipeline = Pipeline(stream)
        >>> list(pipeline)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> pipeline = pipeline.add_filter(negate(is_even))
        >>> pipeline = pipeline.add_map(quadruple)
        >>> list(pipeline)
        [4, 12, 20, 28, 36]

        We can subsample the elements in the pipeline randomly or by providing
        indices to keep.
        >>> stream = range(100)
        >>> pipeline = Pipeline(stream)
        >>> pipeline = pipeline.subsample(10)
        >>> len(list(pipeline))
        10

        We raise a warning if filtering after subsampling.
        >>> try:
        ...     pipeline = pipeline.add_filter(is_even)
        ... except:
        ...     Warning
        ... else:
        ...     print("Filter after subsample warning failed.")
        <class 'Warning'>

        >>> stream = range(100)
        >>> pipeline = Pipeline(stream)
        >>> pipeline = pipeline.add_filter(is_even)
        >>> pipeline = pipeline.subset([1, 2, 3])
        >>> list(pipeline)
        [2, 4, 6]
    """

    def __init__(self, stream, steps=None, precompute_len=False):
        """Initialize pipeline with stream.

        Args:
            stream (iterable): data to iterate over.
            steps (list of (applier, to_apply)): List of manipulations to apply.
            precompute_len (bool): Whether to compute the length at initialization.
        """
        self._stream = stream
        if steps is None:
            steps = []
        self._steps = steps
        self._prev_subsampled = False

        # Using cached_property to compute and store length of stream.
        if precompute_len:
            len(self)

    def __iter__(self):
        stream = self._stream
        for applier, to_apply in self._steps:
            stream = applier(to_apply, stream)

        return (elm for elm in stream)

    def add_filter(self, to_apply):
        """Add a filter step to the pipeline.

        Args:
            to_apply (function): Function to filter on.
        """
        # Scrap the length if it exists as the length has now changed.
        try:
            del self._len
        except AttributeError:
            pass

        # Check whether the stream has been previously subsampled.
        if self._prev_subsampled:
            raise Warning(
                "Filtering after subsampling will lead to further reduction to the number of elements."
            )
        self._steps.append((filter, to_apply))
        return self

    def add_map(self, to_apply):
        """Add a map step to the pipeline.

        Args:
            to_apply (function): Function to map on.
        """
        self._steps.append((map, to_apply))
        return self

    def subsample(self, num_samples):
        """Randomly subsample the elements in the stream."""
        if len(self) > num_samples:
            self._prev_subsampled = True
            keep_idxs = list(
                np.random.choice(len(self), size=num_samples, replace=False)
            )
            self.subset(keep_idxs)
        else:
            warnings.warn("Fewer elements than samples")
        return self

    def subset(self, keep_idxs):
        """Keep only the elements indicated in keep_idxs from the stream."""

        def _applier(to_apply, stream):
            return to_apply(stream)

        keep_idxs = set(keep_idxs)

        def _keep_idxs(stream, keep_idxs=keep_idxs):
            return (item for i, item in enumerate(stream) if i in keep_idxs)

        self._steps.append((_applier, _keep_idxs))
        self._len = keep_idxs
        return self

    @cached_property
    def _len(self):
        return sum(1 for _ in self)

    def __len__(self):
        """Compute the number of items emitted from the pipeline.

        Returns:
            int: Total number of items.
        """
        return self._len

    def __str__(self):
        info_strs = []
        info_strs.append("Initial stream: {}".format(self._stream))
        info_strs.append("Steps to apply:")
        for applier, to_apply in self._steps:
            info_strs.append("\t{} on {}".format(applier, to_apply))
        return "\n".join(info_strs)


def negate(f):
    """Create the boolean inverse of a function.

    Args:
        f (function): The original boolean function

    Returns:
        function: The inverse function. Analogous to `g = lambda x: not f(x)`.

    Example:
        >>> def f(): return True
        >>> g = negate(f)
        >>> g()
        False


        >>> def identity(arg_bool): return arg_bool
        >>> inverse = negate(identity)
        >>> inverse(True)
        False
    """

    @wraps(f)
    def g(*args, **kwargs):
        return not f(*args, **kwargs)

    return g


def listify_nested_iterables(iterable):
    """Recursively convert iterables to lists.

    Recursively exhaust iterables into lists. Careful if you have iterables you
    do not want to accidentally listify. We only handle strings.

    Args:
        iterable (iterable): An iterable whose elements may be other iterables.

    Returns:
        list: A list whose elements

    Examples:
        >>> iterable = iter(["a", "b", "c"])
        >>> listify_nested_iterables(iterable)
        ['a', 'b', 'c']

        >>> iterable = iter(["abc", iter(["def", "ghi"])])
        >>> listify_nested_iterables(iterable)
        ['abc', ['def', 'ghi']]
    """
    nested_list = list(iterable)
    for i, item in enumerate(nested_list):
        # If item is iterable but not a string
        if isinstance(item, Iterable) and not isinstance(item, str):
            nested_list[i] = listify_nested_iterables(item)

    return nested_list


def keep_keys(new_keys, old_dict):
    """Return dictionary with items indicated by the keys.

    Args:
        new_keys (iterable): Keys to keep from the old dictionary.
        old_dict (dict): A dictionary from which to extract a subset of items.

    Returns
        dict: A dict derived from old_dict only keeping keys from new_keys.

    Example:
        To use `keep_keys` directly on a dictionary:

        >>> keep_keys(["a", "b"], {"a": 1, "b": 2, "c": 3})
        {'a': 1, 'b': 2}

        If the requested keys are not present, they are ignored.

        >>> keep_keys(["a", "b"], {"b": 2, "c": 3})
        {'b': 2}

        To use `keep_keys` on a stream of dictionaries:

        >>> dict_gen = iter([{"a": 1, "b": 2, "c": 3},
        ...                  {"b": 5, "c": 6}])
        >>> from functools import partial
        >>> subdict_gen = map(partial(keep_keys, ["a", "b"]), dict_gen)
        >>> list(subdict_gen)
        [{'a': 1, 'b': 2}, {'b': 5}]
    """
    new_dict = {k: old_dict[k] for k in new_keys if k in old_dict}
    return new_dict


def to_proportion(x):
    """Converts values to their proportion.

    Args:
        x (1d array): non-negative values.

    Examples:
        >>> import numpy as np
        >>> x = np.array([ 0, 1, 2, 3, 4])
        >>> to_proportion(x)
        array([0. , 0.1, 0.2, 0.3, 0.4])
    """
    if any(x < -0.0000001):
        raise Exception("Negative values recieved in to_proportion.")
    return x / sum(x)


def sliding_average(data, window=24, axis=1):
    """Average data over sliding window.

    Args:
        data (ndarray): data to average with dimensions: msrmts x num_samples.
        window (int): size of the sliding window to average over.

    Example:
    >>> import numpy as np
    >>> data = np.arange(24).reshape((4,6))
    >>> sliding_average(data, window=5)
    array([[ 2.,  3.],
           [ 8.,  9.],
           [14., 15.],
           [20., 21.]])

    >>> sliding_average(data, window=2, axis=0)
    array([[ 3.,  4.,  5.,  6.,  7.,  8.],
           [ 9., 10., 11., 12., 13., 14.],
           [15., 16., 17., 18., 19., 20.]])

    An exception is raised if there is insufficient data to average over.
    >>> import numpy as np
    >>> data = np.arange(24).reshape((4,6))
    >>> avgd = sliding_average(data, window=10)
    Traceback (most recent call last):
    ...
    Exception: Not enough data to average over with window of size 10.
    """
    if data.shape[axis] < window:
        raise Exception(
            "Not enough data to average over with window of size {}.".format(window)
        )

    # Move axis to be averaged over to fixed position.
    data = np.swapaxes(data, axis, 0)

    # Make a copy to store averaged data (Could alternatively do this in place).
    avgd = np.zeros((data.shape[0] - window + 1, *data.shape[1:]))

    # Average over sliding window.
    for i in range(avgd.shape[0]):
        avgd[i] = np.mean(data[i : i + window], axis=0)

    # Return axis to its original position.
    avgd = np.swapaxes(avgd, axis, 0)

    return avgd


def condense_topic_keywords(topics_freqs, tolerance=1):
    """Remove bigram parts from top keywords by updating frequencies.

    Args:
        topics_freqs (list of dictionaries):
            Association strength between term and topic.

    Returns:
        topics_freqs (list of dictionaries)

    Example:
    >>> test = [{'a': 1,   'b': 3, 'a b': 2},
    ...         {'b': 1, 'a c': 1,   'c': 5},
    ...         {'a': 1,   'b': 1,   'c': 1}]
    >>> condense_topic_keywords(test, tolerance=1)
    [{'a': 0, 'b': 0, 'a b': 3}, {'b': 1, 'a c': 1, 'c': 5}, {'a': 1, 'b': 1, 'c': 1}]
    """
    topics_freqs = deepcopy(topics_freqs)

    for k, topic_freqs in enumerate(topics_freqs):
        for (term, freqs) in topic_freqs.items():
            # Drop terms which have no english letters.
            if not re.match(r".*[a-zA-Z].*", term):
                topic_freqs[term] = 0
                continue

            # Drop unigrams if bigram is nearly as important.
            bigrams = [term for term in topic_freqs if len(term.split()) == 2]
            for bigram in bigrams:
                bigram_freq = topic_freqs[bigram]
                unigrams = bigram.split()
                for unigram in unigrams:
                    # Nothing to do if unigram is not in the topic
                    if unigram not in topic_freqs:
                        continue

                    if topic_freqs[unigram] < (1 + tolerance) * topic_freqs[bigram]:
                        bigram_freq = max(topic_freqs[unigram], bigram_freq)
                        topic_freqs[unigram] = 0

                topic_freqs[bigram] = bigram_freq

    return topics_freqs
