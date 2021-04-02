"""Functions for collecting files of tweet data."""

import datetime
import glob
import os
import re  # For regex.
from datetime import timedelta


def valid_date(date):
    """Check whether the date has the required form of 2020-MM-DD-HH.

    Note that this will still return true if the numeric entries are not valid,
    for example, '2020-45-23-32'.
    """
    # TODO: use glob to check whether the data exists in a file, i.e. is valid.
    date_template = re.compile(r"2020-\d{2}-\d{2}-\d{2}")
    if date_template.fullmatch(date):
        return True
    else:
        return False


def extract_date_hour(filename):
    """Extract the first match of the form 2020-MM-DD-HH in the filename.

    Examples:
    >>> extract_date_hour('coronavirus-tweet-id-2020-03-01-00.txt')
    '2020-03-01-00'

    Print so that the None returned is visible.

    >>> print(extract_date_hour('coronavirus-tweet-id-2020-03-01.txt'))
    None

    >>> extract_date_hour('2020-03-01-00-05')
    '2020-03-01-00'

    >>> extract_date_hour('2020-03-01-00-other_stuff_2020-15-01-00')
    '2020-03-01-00'
    """
    # TODO: return match object to account for multiple dates in file.
    date = re.compile(r"2020-\d{2}-\d{2}-\d{2}")
    date_match = date.search(filename)
    if date_match:
        return date_match.group(0)
    return None


def datetime_from_date_hour(date_hour):
    """Return a datetime object from a date_hour string.

    Example:
    >>> datetime_from_date_hour('2020-01-03-00')
    datetime.datetime(2020, 1, 3, 0, 0)
    """
    return datetime.datetime(*[int(time) for time in date_hour.split("-")])


def year_month_from_date_hour(date_hour):
    """Extract year and month from date_hour.

    Example:
    >>> year_month_from_date_hour('2020-01-03-00')
    '2020-01'
    """
    # TODO: convert to datetime first.
    return date_hour[:7]


def in_range(low_date_hour=None, high_date_hour=None):
    """Return function to determine whether file is in the desired range.

    Example:
    >>> filenames = ['coronavirus-tweet-id-2020-03-01-00.txt',
    ...              'coronavirus-tweet-id-2020-03-04-00.txt',
    ...              'coronavirus-tweet-id-2020-03-02-00.txt',
    ...              'coronavirus-tweet-id-2020-03-00-00.txt',
    ...              'coronavirus-tweet-id-2020-02-04-00.txt',
    ...              'hello']
    >>> filenames = filter(in_range('2020-03-00-00', '2020-03-02-00'),
    ...                    filenames)
    >>> list(filenames)  #doctest: +NORMALIZE_WHITESPACE
    ['coronavirus-tweet-id-2020-03-01-00.txt',
    'coronavirus-tweet-id-2020-03-02-00.txt',
    'coronavirus-tweet-id-2020-03-00-00.txt']

    Test without low_date_hour.
    >>> filenames = ['coronavirus-tweet-id-2020-03-01-00.txt',
    ...              'coronavirus-tweet-id-2020-03-04-00.txt',
    ...              'coronavirus-tweet-id-2020-03-02-00.txt',
    ...              'coronavirus-tweet-id-2020-03-00-00.txt',
    ...              'coronavirus-tweet-id-2020-02-04-00.txt',
    ...              'hello']
    >>> filenames = filter(in_range(high_date_hour='2020-03-02-00'),
    ...                    filenames)
    >>> list(filenames)  #doctest: +NORMALIZE_WHITESPACE
    ['coronavirus-tweet-id-2020-03-01-00.txt',
    'coronavirus-tweet-id-2020-03-02-00.txt',
    'coronavirus-tweet-id-2020-03-00-00.txt',
    'coronavirus-tweet-id-2020-02-04-00.txt']

    Test without high_date_hour.
    >>> filenames = ['coronavirus-tweet-id-2020-03-01-00.txt',
    ...              'coronavirus-tweet-id-2020-03-04-00.txt',
    ...              'coronavirus-tweet-id-2020-03-02-00.txt',
    ...              'coronavirus-tweet-id-2020-03-00-00.txt',
    ...              'coronavirus-tweet-id-2020-02-04-00.txt',
    ...              'hello']
    >>> filenames = filter(in_range(low_date_hour='2020-03-00-00'),
    ...                    filenames)
    >>> list(filenames)  #doctest: +NORMALIZE_WHITESPACE
    ['coronavirus-tweet-id-2020-03-01-00.txt',
    'coronavirus-tweet-id-2020-03-04-00.txt',
    'coronavirus-tweet-id-2020-03-02-00.txt',
    'coronavirus-tweet-id-2020-03-00-00.txt']
    """

    def _in_range(filepath):
        """Indicate whether file is in the desired range."""
        # Get the date in form 2020-MM-DD-HH from file if present, None if not.
        file_date_hour = extract_date_hour(filepath)
        if not file_date_hour:  # No date-hour in file
            return False
        if low_date_hour and file_date_hour < low_date_hour:  # Too early.
            return False
        if high_date_hour and file_date_hour > high_date_hour:  # Too late.
            return False
        return True

    return _in_range


def files_in_range(data_dir, data_range_low, data_range_high):
    """Return files in indicated date-time range."""
    # Check that the date range provided has the required format.
    if not valid_date(data_range_low) or not valid_date(data_range_high):
        raise Exception("Invalid date ranges provided.")

    # Files from which to collect tweets.
    files = glob.glob(os.path.join(data_dir, "2020*/*.gz"), recursive=True)
    # Restrict files to desired range.
    files = filter(in_range(data_range_low, data_range_high), files)
    return files


def filename_to_datetime(filename):
    date_string = extract_date_hour(filename)
    return datetime.datetime.strptime(date_string, "%Y-%m-%d-%H")


def week_from_filename(filename, start_date_hour):
    """Number of weeks passed since start_date_hour for filename.

    >>> filename = 'coronavirus-tweet-id-2020-03-04-05.txt'
    >>> start_date_hour = "2020-03-01-00"
    >>> week_from_filename(filename, start_date_hour)
    0
    >>> filename = 'coronavirus-tweet-id-2020-03-20-05.txt'
    >>> start_date_hour = "2020-03-01-00"
    >>> week_from_filename(filename, start_date_hour)
    2
    """
    time_since_start = filename_to_datetime(filename) - datetime.datetime.strptime(
        start_date_hour, "%Y-%m-%d-%H"
    )
    return int(time_since_start / timedelta(days=7))


def day_from_filename(filename, start_date_hour):
    """Number of days passed since start_date_hour for filename.

    >>> filename = 'coronavirus-tweet-id-2020-03-04-05.txt'
    >>> start_date_hour = "2020-03-01-00"
    >>> day_from_filename(filename, start_date_hour)
    3
    """
    time_since_start = filename_to_datetime(filename) - datetime.datetime.strptime(
        start_date_hour, "%Y-%m-%d-%H"
    )
    return int(time_since_start / timedelta(days=1))
