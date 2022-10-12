""" Calculates metrics and generate reports """

from typing import Union, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


class Metric:
    """
    Object that calculates metrics and generates reports
    """

    def __init__(self, name: str, y_pred: Union[np.ndarray, pd.Series] = None,
                 y_true: Union[np.ndarray, pd.Series] = None, multiclass: bool = False) -> None:
        """
        Class constructor

        Parameters
        ----------
        name : str
            Name (to be used on reports)
        y_pred : ND array or DataFrame
            An array with prediction of a models
        y_true : ND array or DataFrame
            The expected label
        multiclass : bool, optional
            If there are multiple class
        """
        self.__name__ = name
        if y_pred is None or y_true is None:
            return
        average = "binary"
        if multiclass:
            average = None
        self.__acc__ = accuracy_score(y_true, y_pred)
        self.__pre__ = precision_score(y_true, y_pred, average=average, zero_division=0)
        self.__f1__ = f1_score(y_true, y_pred, average=average)
        self.__recall__ = recall_score(y_true, y_pred, average=average)
        self.__cm__ = confusion_matrix(y_true, y_pred)
        return

    def accuracy(self) -> float:
        """
        Return accuracy value

        Returns
        -------
            A float value indicating accuracy of test
        """
        return self.__acc__

    def precision(self) -> np.ndarray:
        """
        Return precision values

        Returns
        -------
        An array with precision for every class
        """
        return self.__pre__

    def f1_score(self) -> np.ndarray:
        """
        Return f1 score values

        Returns
        -------
        An array with f1-score for every class
        """
        return self.__f1__

    def recall(self) -> np.ndarray:
        """
        Return recall values

        Returns
        -------
        An array with recall for every class
        """
        return self.__recall__

    def confusion_matrix(self) -> np.ndarray:
        """
        Return recall values

        Returns
        -------
        An array with recall for every class
        """
        return self.__cm__

    def add_to_report(self, report: pd.DataFrame, labels: List[str] = None) -> pd.DataFrame:
        """
        Add metrics to report on a DataFrame

        Parameters
        ----------
        report : DataFrame
            DataFrame to add metrics
        labels : List[str], optional
            Name of classes. If None given it assumes binary

        Returns
        -------
        DataFrame :
            DataFrame altered
        """
        index = self.__name__
        new_row = pd.DataFrame([[0] * len(report.columns)], columns=report.columns, index=[index])
        new_row.loc[index, "Accuracy"] = self.accuracy()
        try:
            candid_labels = len(self.precision())
            for metric, values in zip(
                    ["Precision", "Recall", "f1-Score"],
                    [self.precision(), self.recall(), self.f1_score()]
            ):
                if labels is None:
                    labels = range(candid_labels)
                metric_columns = ["{} {}".format(metric, label) for label in labels]
                new_row.loc[index, metric_columns] = values
        except TypeError:
            new_row.loc[index, "Precision"] = self.precision()
            new_row.loc[index, "Recall"] = self.recall()
            new_row.loc[index, "f1-Score"] = self.f1_score()
        return pd.concat((report, new_row), axis=0)
