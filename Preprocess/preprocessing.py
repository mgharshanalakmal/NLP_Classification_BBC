from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

"""
Transformer to encode strings to numerical values.
"""


class TextToSequence(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer) -> None:
        super(TextToSequence, self).__init__()
        self.tokenizer = tokenizer

    def fit(self, X: pd.DataFrame, y):
        return self

    def transform(self, X: pd.DataFrame):
        text_sequences = self.tokenizer.texts_to_sequences(X)

        return text_sequences


"""
Transformer to padding strings to predefined length.
"""


class PadSequences(BaseEstimator, TransformerMixin):
    def __init__(self, max_length) -> None:
        super(PadSequences, self).__init__()
        self.max_length = max_length

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_padded = pad_sequences(X, maxlen=self.max_length, padding="post", truncating="post")

        return X_padded
