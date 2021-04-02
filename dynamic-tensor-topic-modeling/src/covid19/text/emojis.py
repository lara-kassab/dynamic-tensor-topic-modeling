"""Functions for extracting and manipulating emojis."""

import demoji

demoji.download_codes()


def extract_emojis(text):
    """Return a list of emojis contained in text.

    The order and counts of the emojis are preserved.
    Examples:
    >>> extract_emojis("🦠 coronavirus. 🤒😷😷😷")
    ['🦠', '🤒', '😷', '😷', '😷']
    >>> extract_emojis("No emojis here. :( )")
    []
    """
    return [emoji for emoji in demoji._EMOJI_PAT.findall(text)]


def contains_emoji(text):
    """Return True if text contains an emoji, False otherwise.

    Examples:
    >>> contains_emoji("🦠 coronavirus. 🤒😷😷😷")
    True
    >>> contains_emoji("No emojis here. :( )")
    False
    """
    return bool(demoji.findall(text))
