""" Model Abstract Class """

from __future__ import annotations
from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Model Abstract class
    """

    @abstractmethod
    def __init__(self, name: str) -> None:
        """
        Class constructor

        Parameters
        ----------
        name : str
            A String indicating name (will use to make reports)
        """
        self.__name__ = name
        self.__model__ = None
        self.__trained_data__ = None
        return

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: Union[np.ndarray, pd.Series], **kwargs) -> Model:
        """

        Parameters
        ----------
        x_train : DataFrame
            Data to train model
        y_train : ND array or DataFrame
            Labels to classify

        Other Parameters
        ----------------
        Parameters for particular model during training

        Returns
        -------
        Model:
            Trained Model
        """
        return Model("empty")

    @abstractmethod
    def classify(self, x_test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Classifies test data, giving the probability
        of output class

        Parameters
        ----------
        x_test : DataFrame
            The test data

        Returns
        -------
        DataFrame:
            Probabilities of each class
        """
        return pd.DataFrame([])

    @abstractmethod
    def predict(self, x_test: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Predicts test data and return the prediction

        Parameters
        ----------
        x_test : DataFrame
            The test data

        Returns
        -------
        Series
            Predicted Labels
        """
        return pd.Series([])

    @abstractmethod
    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return dict()

    def save_model(self, path: str) -> None:
        """
        Saves model to disk

        Parameters
        ----------
        path : str
            Path to save model

        Returns
        -------
        None
        """
        return

    def load_model(self, path: str) -> None:
        """
        Loads model to disk

        Parameters
        ----------
        path : str
            Path to save model

        Returns
        -------
        None
        """
        return
