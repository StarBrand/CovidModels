""" Holdout trainer """

import os
import gc
import numpy as np
import pandas as pd
from typing import List, Tuple
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from accessories.trainer import Trainer
from accessories import Metric
from models import Model
from accessories.config import CONFIG


class HoldoutTrainer(Trainer):
    """
    Holdout Trainer, trains a model
    by using holdout (split test) validation

    Attributes
    ----------
    __fpr__ : NDArray
        False positive rates
    __tpr__ : NDArray
        True positive rates
    __auc__ : float
        Area Under the Curve
    """
    def __init__(self, multiclass: bool = False, labels: List[str] = None) -> None:
        super(HoldoutTrainer, self).__init__(multiclass=multiclass, labels=labels)
        self.__fpr__: np.ndarray = np.array(list())
        self.__tpr__: np.ndarray = np.array(list())
        self.__auc__: float = 0.0
        return

    def train_model(self, model: Model, x: pd.DataFrame, y: pd.Series, resampling: str = None, **kwargs) -> None:
        """
        Trains model using holdout validation

        Parameters
        ----------
        model : Model
            Model to be trained by holdout
        x : DataFrame
            Dataset
        y : Series
            Labels
        resampling : str, optional
            Resampling method, valid "under", "over" or None


        Other Parameters
        ----------------
        path_to_model : str, optional
            If given, saves the model to path
        test_size : float, optional
            Portion of dataset to use as test
        plot_validation : bool, optional
            If plot testing
        kwargs
            Other parameters of model.train method

        Returns
        -------
        None
        """
        self.__training_data__.clear()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=kwargs.get("test_size", CONFIG["TEST_SIZE"]))
        if resampling is not None:
            if resampling == "under":
                x_train, y_train = self.__undersample__(x_train, y_train)
            elif resampling == "over":
                x_train, y_train = self.__oversample__(x_train, y_train)
            else:
                raise AttributeError("Sampling method {} not valid".format(resampling))
        val_set = kwargs.get("plot_validation", False)
        if val_set:
            x_val, x_test, y_val, y_test = train_test_split(
                x_test, y_test, test_size=0.5
            )
            if "path_to_model" in kwargs.keys():
                path_to_model = os.path.join(kwargs.get("path_to_model"), model.__name__)
                kwargs.pop("path_to_model")
                trained_model = model.train(x_train, y_train, val_set=(x_val, y_val),
                                            path_to_model=path_to_model,
                                            **kwargs)
            else:
                trained_model = model.train(x_train, y_train, val_set=(x_val, y_val), **kwargs)
        else:
            trained_model = model.train(x_train, y_train, **kwargs)
        if "path_to_model" in kwargs.keys():
            trained_model.save_model(os.path.join(
                kwargs.get("path_to_model"),
                trained_model.__name__
            ))
        y_pred = trained_model.predict(x_test, **kwargs)
        probabilities = trained_model.classify(x_test, **kwargs)
        self.__metric__ = Metric(model.__name__, y_pred, y_test, multiclass=self.__multi__)
        if self.__multi__:
            true_prob = pd.Series(dtype="float64")
            for i in probabilities.columns:
                true_prob = pd.concat([true_prob, (y_test == i) * 1.0])
            self.__fpr__, self.__tpr__, _ = roc_curve(true_prob, probabilities.to_numpy().ravel())
        else:
            self.__fpr__, self.__tpr__, _ = roc_curve(y_test, probabilities[1])
        self.__auc__ = auc(self.__fpr__, self.__tpr__)
        self.__training_data__ = trained_model.train_data()
        del x_train, x_test, trained_model
        gc.collect()
        return

    def plot_roc_curve(self, axes: Axes) -> Axes:
        """
        Plots Receiver Operating Characteristic (ROC) curve
        in Holdout Validation Context

        Adapted from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

        Parameters
        ----------
        axes : Axes
            Axes to plot

        Returns
        -------
        Axes
            Axes with plot added
        """
        disclaimer = ""
        if self.__multi__:
            disclaimer = "micro-average "
        axes.plot(
            self.__fpr__, self.__tpr__, alpha=0.8, lw=1, color="blue",
            label="{}ROC curve (AUC = {:.2f})".format(disclaimer, self.__auc__)
        )
        axes.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        return super(HoldoutTrainer, self).plot_roc_curve(axes)

    def plot_training(self,
                      metric_axes: Axes,
                      loss_axes: Axes,
                      metric_val_axes: Axes,
                      loss_val_axes: Axes,
                      **kwargs) -> Tuple[Axes, Axes, Axes, Axes]:
        """
        Plots training process

        Parameters
        ----------
        metric_axes : Axes
            matplotlib.axes.Axes where to plot training metric
        loss_axes : Axes
            matplotlib.axes.Axes where to plot training loss
        metric_val_axes : Axes
            matplotlib.axes.Axes where to plot testing metric
        loss_val_axes : Axes
            matplotlib.axes.Axes where to plot testing loss

        Other Parameters
        ----------------
        append : bool
            If there is going to be several neural network on graphs
        last_one : bool
            If append, whether or not this is is the last model to append

        Returns
        -------
        Axes
            Axes with training metric plotted
        Axes
            Axes with training loss plotted
        Axes
            Axes with validation metric plotted
        Axes
            Axes with validation loss plotted
        """
        append = kwargs.get("append", False)
        metric_axes.plot(self.__training_data__["Learning metric"],
                         label=self.__metric__.__name__ if append else None)
        loss_axes.plot(self.__training_data__["Loss"],
                       label=self.__metric__.__name__ if append else None)
        metric_val_axes.plot(self.__training_data__["Testing metric"],
                             label=self.__metric__.__name__ if append else None)
        p = loss_val_axes.plot(self.__training_data__["Test Loss"],
                               label=self.__metric__.__name__ if append else None)
        if len(self.__training_data__["Early stops"]) > 0:
            for epoch, loss in self.__training_data__["Early stops"]:
                loss_val_axes.annotate("", xy=(epoch, loss), xycoords="data",
                                       xytext=(0, 25), textcoords="offset points",
                                       arrowprops=dict(facecolor=p[-1].get_color(), shrink=0.05))
        if append:
            last_one = kwargs.get("last_one", False)
        else:
            last_one = True
        if last_one:
            return super(HoldoutTrainer, self)._postprocessing_training(
                (metric_axes, loss_axes, metric_val_axes, loss_val_axes), legend=append
            )
        else:
            return metric_axes, loss_axes, metric_val_axes, loss_val_axes

    def plot_feature_selection(self, axes: Axes, feature_selection_path: str = None) -> Axes:
        """
        Plots feature selection from models in with feature importance (or coefficient)

        Parameters
        ----------
        axes : Axes
            Axes to plot feature selection
        feature_selection_path : str, optional
            Path to save feature selection

        Raises
        ------
        AttributeError
            Model not support feature importance

        Returns
        -------
        Axes
            Axes with plot on it
        """
        if "Feature Importance" not in self.__training_data__.keys():
            raise AttributeError("Model not support feature importance")
        feature_importance = pd.DataFrame(
            self.__training_data__["Feature Importance"],
            columns=["Feature Importance"], index=self.__training_data__["Features"]
        )
        mask = feature_importance["Feature Importance"] > self.__training_data__["Threshold"]
        feature_importance = feature_importance.loc[
            feature_importance[mask].index, :
        ]
        feature_importance.sort_values("Feature Importance", ascending=True, inplace=True)
        if feature_selection_path is not None:
            feature_importance["Variable"] = feature_importance.index.str.split("_").str[0]
            feature_importance[["Variable", "Feature Importance"]].to_csv(
                feature_selection_path,
                header=True, index=True, sep=","
            )
        return super(HoldoutTrainer, self).__plot_feature_selection__(
            self.__metric__.__name__ + "\n", axes, feature_importance["Feature Importance"]
        )
