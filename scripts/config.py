"""Load local configuration parameters."""

# fmt: off
example = """
    Example contents of _local_config.py:
    data_dir = "/local/path/to/COVID-19-TweetIDs"
    results_dir = "/local/path/to/results_dir"

    These directories are:
    data_dir: the local path to the data directory.
    results_dir: the local path to the desired results directory.
    """

try:
    from _local_config import data_dir, results_dir  # noqa: F401
except ModuleNotFoundError:
    print(
    """
    _local_config.py not found. Create _local_config.py in the scripts
    directory.
    """ + example)  # noqa: E122
    raise

except ImportError:
    print(
    """
    Missing required variable data_dir or results_dir.
    """ + example)  # noqa: E122
    raise
# fmt: on
