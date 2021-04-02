"""Functions for cleaning tweet['full_text']."""

import re  # For regex processing.


def remove_links(text):
    """Remove links/urls from text.

    Twitter automatically shortens all links to https://t.co/<hash>. We
    take advantage of this fact to strip links with a very simple regex.

    Args:
        text (str): Text from which to remove links.

    Returns:
        str: The same text with links such as "https://t.co/FakeLink" removed.

    Examples:
        >>> remove_links('No links here.')
        'No links here.'

        >>> remove_links(r'There was a link: https://t.co/FakeLink')
        'There was a link: '
    """
    return re.sub(r"http[s]?://\S+", "", text)


def remove_user_tags(text):
    """Remove user tags such as @realDonaldTrump from text.

    Args:
        text (str): Text from which to remove links.

    Returns:
        str: The same text with user tags such as @realDonaldTrump removed.

    Notes: Regex from https://stackoverflow.com/a/6351873

    Examples:
        >>> remove_user_tags('@user test text')
        ' test text'

        >>> remove_user_tags('email@user.com')
        'email@user.com'

        >>> remove_user_tags('test text')
        'test text'
    """
    return re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", "", text)
