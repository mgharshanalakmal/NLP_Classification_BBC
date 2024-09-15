from Preprocess.preprocessing import TextToSequence, PadSequences

from typing import Any, List, Dict
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import keras
import numpy as np
import matplotlib.pyplot as plt


class DeepLearningModel(object):
    METRICS = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    BATCH_SIZE = 100
    EPOCHS = 5

    def __init__(self, dl_model: Any, X_train: List, Y_train: List, X_test: List, Y_test: List, padding_length: int, label_dictionary: Dict) -> None:
        self.dl_model = dl_model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.padding_length = padding_length
        self.label_dictionary = label_dictionary

        self.unique_word_count: int = self.unique_words()
        self.fit_tokernizer()

        self.preprocessing_pipe: Pipeline = Pipeline(
            steps=[
                ("text_to_sequences", TextToSequence(tokenizer=self.tokenizer)),
                ("pad_sequences", PadSequences(max_length=self.padding_length)),
            ]
        )

        self.processed_X_train = self.preprocessing_pipe.fit_transform(self.X_train, self.Y_train)
        self.processed_X_test = self.preprocessing_pipe.fit_transform(self.X_test, self.Y_test)
        self.Y_train = to_categorical(self.Y_train, num_classes=5)

        self.model_init()

        self.predict_results = self.predict()

    @staticmethod
    def counter_word(text_array):
        count = Counter()
        for text in text_array:
            for word in text.split():
                count[word] += 1
        return count

    def unique_words(self) -> int:
        counter: Counter = self.counter_word(self.X_train)
        num_unique_words: int = len(counter)

        return num_unique_words

    def fit_tokernizer(self) -> None:
        self.tokenizer: Tokenizer = Tokenizer(num_words=self.unique_word_count)
        self.tokenizer.fit_on_texts(self.X_train)

    def model_init(self):
        self.dl_model = self.dl_model(self.unique_word_count, 5)
        self.dl_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=self.METRICS)
        self.dl_model.fit(self.processed_X_train, self.Y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, validation_split=0.2)

    def predict(self):
        predictted_resuts = self.dl_model.predict(self.processed_X_test)
        results = predictted_resuts.argmax(axis=1)

        return results

    def classification_report(self):
        print(classification_report(self.Y_test, self.predict_results, zero_division=0))

    def plot_confution_matrix(self):
        confusion_matrix_ = confusion_matrix(self.Y_test, self.predict_results)

        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix_, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = list(self.label_dictionary.keys())
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(
                    j,
                    i,
                    str(confusion_matrix_[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_matrix_[i, j] > confusion_matrix_.max() / 2 else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()
