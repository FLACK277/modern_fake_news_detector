import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_PATTERN = re.compile(r"http\S+|www\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\S+@\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
HTML_PATTERN = re.compile(r"<.*?>")
NUMBER_PATTERN = re.compile(r"\d+")
TOKEN_PATTERN = re.compile(r"[a-z]{3,}")


def clean_text(text: str) -> str:
    """Notebook-based preprocessing adapted for production use."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = [tok for tok in TOKEN_PATTERN.findall(text) if tok not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def clean_text_batch(texts):
    """Applies notebook-style cleaning to a sequence of texts."""
    return [clean_text(text) for text in texts]
