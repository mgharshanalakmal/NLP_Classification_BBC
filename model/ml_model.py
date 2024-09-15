from typing import Any, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import scipy
import numpy as np
import matplotlib.pyplot as plt


class MLModel(object):
    def __init__(
        self,
        ml_model: Any,
        vecotrizer: TfidfVectorizer,
        x_train: List,
        x_test: List,
        y_train: List,
        y_test: List,
        label_dictionary: Dict,
    ) -> None:
        self.ml_model = ml_model
        self.vectorizer = vecotrizer
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_dictionary = label_dictionary

        self.vectorization()
        self.training()
        self.predict()
        self.scoring()

    def vectorization(self) -> None:
        self.X_train_features: scipy.sparse._csr.csr_matrix = self.vectorizer.fit_transform(self.x_train)
        self.X_test_features: scipy.sparse._csr.csr_matrix = self.vectorizer.transform(self.x_test)

    def training(self) -> None:
        self.ml_model.fit(self.X_train_features, self.y_train)

    def predict(self) -> None:
        self.y_pred = self.ml_model.predict(self.X_test_features)

    def scoring(self) -> None:
        self.scores = {}
        self.classification_report = classification_report(self.y_test, self.y_pred)

        self.scores["Accuracy"] = accuracy_score(self.y_test, self.y_pred)
        self.scores["F1 Score"] = f1_score(self.y_test, self.y_pred, average=None)
        self.scores["Precision"] = precision_score(self.y_test, self.y_pred, average=None, zero_division=0.0)
        self.scores["Recall"] = recall_score(self.y_test, self.y_pred, average=None)

    def classification_report_print(self) -> None:
        print(self.classification_report)

    def confusion_matrix_plot(self):
        confusion_matrix_ = confusion_matrix(self.y_test, self.y_pred)

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
