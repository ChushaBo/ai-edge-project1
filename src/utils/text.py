"""Text helpers for 06_embed_text: masking damage figures and chunking long narratives."""

import hashlib
import re

AMOUNT = "[AMOUNT]"
NUM = "[NUM]"

# Dollar figures: "$1.2 million", "$500K", "$25,000", "10.00K" (NOAA style), "3 million dollars".
_DOLLARS = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand)|[KMB]\b)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)\s+dollars\b"
    r"|\b\d[\d,]*(?:\.\d+)?[KMB]\b",
    re.IGNORECASE)

# Things that get damaged, counted in narratives ("over 1,200 homes", "30 mobile homes").
_DAMAGE_NOUNS = (r"homes?|houses?|residences?|dwellings?|buildings?|structures?|businesses|"
                 r"properties|apartments?|units?|condos?|trailers?|vehicles?|cars?|boats?")
# A count directly before (up to two words between) a damaged-thing noun.
_COUNT_BEFORE_NOUN = re.compile(
    rf"\b\d[\d,]*\+?(?=(?:\s+[A-Za-z-]+){{0,2}}\s+(?:{_DAMAGE_NOUNS})\b)", re.IGNORECASE)
# A count after a damage verb ("damaged 45", "destroyed over 100"); the verb and any
# qualifier are kept (groups 1-2), only the number (group 3) is replaced.
_COUNT_AFTER_VERB = re.compile(
    r"\b(damaged|destroyed|flooded|inundated|affected|impacted)"
    r"(\s+(?:over\s+|about\s+|around\s+|approximately\s+|nearly\s+|more\s+than\s+)?)"
    r"(\d[\d,]*)", re.IGNORECASE)


def mask_damage_numbers(text: str) -> str:
    """Replace dollar amounts with [AMOUNT] and counts of damaged things with [NUM].

    Meteorological numbers (rainfall, wind speed, surge height, dates) are kept: they
    describe the storm, not the loss.
    """
    text = _DOLLARS.sub(AMOUNT, text)
    text = _COUNT_AFTER_VERB.sub(rf"\g<1>\g<2>{NUM}", text)
    return _COUNT_BEFORE_NOUN.sub(NUM, text)


def chunk_words(text: str, max_words: int) -> list[str]:
    """Split text into chunks of at most max_words words, breaking at sentence ends
    where possible, so no chunk exceeds the embedding model's input limit."""
    sentences = re.split(r"(?<=[.!?])\s+|\s*\|\|\s*", text.strip())
    chunks, current = [], []
    for sentence in filter(None, sentences):
        words = sentence.split()
        while len(words) > max_words:  # a single over-long sentence
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        current += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def text_key(model: str, text: str) -> str:
    """Stable cache key for (model, text)."""
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()
