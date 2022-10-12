""" Sklearn models Wrapper """

import os
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from joblib import dump, load
from typing import Union, Dict, Any, List
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from models import Model


class SklearnModel(Model, ABC):
    """
    Sklearn classifier wrapper
    """

    @abstractmethod
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.__model__: BaseEstimator
        self.__features__: List[str] = list()

    def train(self, x_train: pd.DataFrame, y_train: Union[np.ndarray, pd.Series], **kwargs) -> Model:
        """
        Train a model, returning the model trained

        Parameters
        ----------
        x_train : DataFrame
            Input data
        y_train : ND array or DataFrame
            Labels

        Other Parameters
        ----------------
        None

        Returns
        -------
        Model :
            Trained model
        """
        model = deepcopy(self)
        model.__model__.fit(x_train, y_train)
        model.__features__ = list(x_train.columns)
        return model

    def classify(self, x_test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Classify an input

        Parameters
        ----------
        x_test : DataFrame
            Input data

        Returns
        -------
        DataFrame:
            Probabilities of each class
        """
        return pd.DataFrame(self.__model__.predict_proba(x_test), index=x_test.index)

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
        return pd.Series(self.__model__.predict(x_test), index=x_test.index)

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return {
            "Feature Importance": self.__model__.feature_importances_,
            "Features": self.__features__,
            "Threshold": self.__model__.feature_importances_.mean()
        }

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
        logging.info("Saving Model {} to {}".format(self.__name__, os.path.dirname(path)))
        dump(self.__model__, path + ".joblib")
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
        logging.info("Loading Model {} to {}".format(self.__name__, path + ".joblib"))
        self.__model__ = load(path + ".joblib")
        return


class Dummy(SklearnModel):
    """
    Wrapper of DummyClassifier from sklearn
    """
    def __init__(self) -> None:
        super().__init__(name="Dummy Mode Classifier")
        self.__model__ = DummyClassifier(strategy="most_frequent")

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return dict()


class DecisionTree(SklearnModel):
    """
    Decision Tree Classifier from sklearn wrapper
    """
    def __init__(self) -> None:
        super().__init__(name="Decision Tree")
        self.__model__ = DecisionTreeClassifier(criterion="gini", splitter="best")


class RandomForest(SklearnModel):
    """
    Random Forest Classifier from sklearn wrapper
    """
    def __init__(self) -> None:
        super().__init__(name="Random Forest")
        self.__model__ = RandomForestClassifier()


class SVM(SklearnModel):
    """
    Support Vector Machine Classifier from sklearn wrapper
    """
    def __init__(self) -> None:
        super().__init__(name="SVM")
        self.__model__ = LinearSVC(loss="squared_hinge")

    def classify(self, x_test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Classify an input

        Parameters
        ----------
        x_test : DataFrame
            Input data

        Returns
        -------
        DataFrame:
            Probabilities of each class
        """
        return pd.DataFrame({
            0: 1.0 - self.__model__.predict(x_test),
            1: self.__model__.predict(x_test)
        }, index=x_test.index)

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return {
            "Feature Importance": np.abs(self.__model__.coef_[0]),
            "Features": self.__features__,
            "Threshold": np.abs(self.__model__.coef_[0]).mean()
        }


class NaiveBayes(SklearnModel):
    """
    Naive Bayes Classifier from sklearn wrapper
    """
    def __init__(self) -> None:
        super().__init__(name="Naive Bayes")
        self.__model__ = GaussianNB()

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return dict()


class Gaussian(SklearnModel):
    """
    Gaussian Process Classifier from sklearn wrapper
    """
    def __init__(self) -> None:
        super().__init__(name="Gaussian Classifier")
        self.__model__ = GaussianProcessClassifier(optimizer="fmin_l_bfgs_b", max_iter_predict=1000)

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return dict()


class LogReg(SklearnModel):
    """
    Logistic Regression Classifier from sklearn wrapper
    """
    def __init__(self) -> None:
        super().__init__(name="Logistic Regression")
        self.__model__ = LogisticRegression(penalty="l1", solver="saga", max_iter=1000)

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return {
            "Feature Importance": np.abs(self.__model__.coef_[0]),
            "Features": self.__features__,
            "Threshold": 1e-5
        }


class XGBoost(SklearnModel):
    """
    XGBoost Classifier  wrapper
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        super().__init__(name="XGBoost")
        self.__model__ = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

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
        logging.info("Saving Model {} to {}".format(self.__name__, os.path.dirname(path)))
        self.__model__.save_model(path + ".json")
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
        logging.info("Loading Model {} to {}".format(self.__name__, path + ".json"))
        self.__model__.load_model(path + ".json")
        return
