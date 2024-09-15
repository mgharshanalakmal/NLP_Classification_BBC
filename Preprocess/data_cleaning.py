from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from typing import List

import re
import string
import nltk
import pandas as pd


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")


stop = set(stopwords.words("english"))


class LowerCase(BaseEstimator, TransformerMixin):
    """
    Transform tweets in "text" column into simple letters to remove duplication of words on letters.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        return [text.lower() for text in X]


class RemoveURL(BaseEstimator, TransformerMixin):
    """
    Remove URL strings in tweets.
    """

    def __init__(self) -> None:
        super(RemoveURL, self).__init__()
        self.url_pattern = r"https?\s*://(?:\s*\S)+"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        return [re.sub(self.url_pattern, "", text) for text in X]


class RemoveEmojis(BaseEstimator, TransformerMixin):
    """
    Remove emojis in tweets.
    """

    def __init__(self) -> None:
        super(RemoveEmojis, self).__init__()
        self.emoji_patterns = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        return [re.sub(self.emoji_patterns, "", text) for text in X]


class RemovePunctuations(BaseEstimator, TransformerMixin):
    """
    Remove punctuations in tweets.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    @staticmethod
    def remove_punct(text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        return [self.remove_punct(text) for text in X]


class RemoveStopWords(BaseEstimator, TransformerMixin):
    """
    Remove stop words in tweets.
    """

    def fit(self, X, y=None):
        return self

    @staticmethod
    def remove_stopwords(text: str) -> str:
        filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(filtered_words)

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        return [self.remove_stopwords(text) for text in X]
