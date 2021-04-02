import gzip
import os
import pickle

from loguru import logger
from tqdm import tqdm


class LinesFromGzipFiles:
    """Yield lines from gzipped jsonl files.

    Args:
        filepaths (iterable of str): paths to one or more files.

    Yields:
        str: The next line of the file.

    Example:
        >>> lines = LinesFromGzipFiles(["file1.gz", "file2.gz"])
        >>> for line in lines:  # doctest: +SKIP
        ...     process(line)   # doctest: +SKIP
    """

    def __init__(self, filepaths):
        # This empties `filepaths` if it is a generator.
        self._filepaths = list(filepaths)
        self._n_files = len(self._filepaths)
        self._len = None

    def __iter__(self):
        # TODO: Inner tqdm?
        for filepath in tqdm(self._filepaths, leave=False):
            with gzip.open(filepath, "r") as f:
                yield from f  # Yields the lines, one at a time.

    def __len__(self):
        """Compute the number of lines among all of the files.

        Returns:
            int: Total lines in all the files.
        """
        if self._len is None:
            logger.info("Counting the lines in {} files.", self._n_files)
            self._len = sum(1 for _ in self)

        return self._len


def save_in_subdir(data, filename, output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    output_file = os.path.join(output_folder, filename)
    with open(output_file + ".pickle", "wb") as file:
        pickle.dump(data, file)
