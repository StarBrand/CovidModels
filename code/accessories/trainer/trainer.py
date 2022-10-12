""" Trainer superclass """

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from accessories import Metric
from models import Model
from accessories.config import CONFIG, CLINICAL_ATTRIBUTES


class Trainer(ABC):
    """
    Trainer class
    """
    def __init__(self, multiclass: bool = False, labels: List[str] = None) -> None:
        """
        Constructor

        Parameters
        ----------
        multiclass : bool, optional
            To use more than two classes
        labels : List[str]
            Classes of prediction, if None it assumes binary classification or numbers
        """
        self.__metric__: Metric = Metric(name="NULL")
        self.__training_data__ = dict()
        self.__multi__ = multiclass
        self.__labels__ = labels if labels is not None else list()
        return

    @abstractmethod
    def train_model(self, model: Model, x: pd.DataFrame, y: pd.Series, resampling: str = None, **kwargs) -> None:
        """
        Trains model according to Trainer

        Parameters
        ----------
        model : Model
            Model to be trained
        x : DataFrame
            Dataset
        y : Series
            Labels
        resampling : str, optional
            Resampling method, valid "under", "over" or None

        Other Parameters
        ----------------
        Parameters for training model

        Raises
        ------
        AttributeError
            If resampling method is not valid

        Returns
        -------
        None
        """
        return

    def add_to_report(self, report: pd.DataFrame) -> pd.DataFrame:
        """
        Add metrics to report on a DataFrame

        Parameters
        ----------
        report : DataFrame
            DataFrame to add metrics

        Returns
        -------
        DataFrame :
            DataFrame altered
        """
        if len(self.__labels__) != 0:
            return self.__metric__.add_to_report(report, labels=self.__labels__)
        else:
            return self.__metric__.add_to_report(report)

    def plot_confusion_matrix(self, axes: Axes, labels: List[str]) -> Axes:
        """
        Plots Confusion matrix in axes

        Parameters
        ----------
        axes : Axes
            Axes to plot confusion matrix
        labels : List[str]
            Labels name

        Returns
        -------
        Axes
            Axes with confusion matrix
        """
        cm = self.__metric__.confusion_matrix()
        name = self.__metric__.__name__
        axes.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            axes.text(j, i, z, ha="center", va="center", fontsize=CONFIG["LABEL_SIZE"])
        axes.set_xlabel("Predicted\n", fontsize=CONFIG["LABEL_SIZE"])
        axes.set_ylabel("True\n", fontsize=CONFIG["LABEL_SIZE"])
        axes.set_title(name + "\n", fontsize=CONFIG["TITLE_SIZE"])
        axes.tick_params(which="both", bottom=False)
        axes.xaxis.set_label_position('top')
        axes.xaxis.set_ticks(range(0, cm.shape[0]))
        axes.set_xticklabels(labels, fontsize=CONFIG["TICK_SIZE"])
        axes.yaxis.set_ticks(range(0, cm.shape[1]))
        axes.set_yticklabels(labels, fontsize=CONFIG["TICK_SIZE"])
        return axes

    @abstractmethod
    def plot_roc_curve(self, axes: Axes) -> Axes:
        """
        Plots ROC curve

        Parameters
        ----------
        axes : Axes
            Axes to plot ROC curve

        Returns
        -------
        Axes
            Axes with new plot
        """
        axes.legend(fontsize=CONFIG["LEGEND_SIZE"])
        axes.set_xlabel("False Positive Rate", fontsize=CONFIG["LABEL_SIZE"])
        axes.set_ylabel("True Positive Rate", fontsize=CONFIG["LABEL_SIZE"])
        axes.set_title(self.__metric__.__name__ + "\n", fontsize=CONFIG["TITLE_SIZE"])
        return axes

    @abstractmethod
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
        kwargs
            Arguments of child implementations

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
        pass

    @staticmethod
    def _postprocessing_training(axes: Tuple[Axes, Axes, Axes, Axes], legend: bool = True) -> Tuple[Axes, Axes, Axes, Axes]:
        """
        Add labels, legends and title to Learning curve plots

        Parameters
        ----------
        axes : Tuple[Axes]
            All axes to label, the order is as plot_training
        legend : bool, optional
            Add legend to graph, default true

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
        for i, _ in enumerate(axes):
            axes[i].set_xlabel("Epoch", fontsize=CONFIG["LABEL_SIZE"])
            if (i % 2) == 0:
                axes[i].set_ylabel("F1-score", fontsize=CONFIG["LABEL_SIZE"])
                title = "{} metric"
            else:
                axes[i].set_ylabel("Loss", fontsize=CONFIG["LABEL_SIZE"])
                title = "{} loss"
            if i < 2:
                axes[i].set_title(title.format("Training"), fontsize=CONFIG["TITLE_SIZE"])
            else:
                axes[i].set_title(title.format("Validation"), fontsize=CONFIG["TITLE_SIZE"])
            if legend:
                axes[i].legend(fontsize=CONFIG["LEGEND_SIZE"])
        return axes

    @abstractmethod
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
        pass

    @staticmethod
    def __plot_feature_selection__(name: str, axes: Axes, feature_importance: pd.Series, error: pd.Series = None) -> Axes:
        """
        Plots feature selection

        Parameters
        ----------
        name : str
            Name of model
        axes : Axes
            Axes to plot feature selection
        feature_importance : Series
            Feature importance of selecting variance
        error : Series, optional
            Standard deviation of feature importance

        Returns
        -------
        Axes
            Axes with plot on it
        """
        features = list()
        colors = list()
        labels = list()
        patches_handles = list()
        clinical_categories = dict()
        for category in CLINICAL_ATTRIBUTES:
            for attribute in CLINICAL_ATTRIBUTES[category]["attributes"]:
                clinical_categories[attribute] = category
        for index in feature_importance.index:
            features.append(index.replace("_", ":\n"))
            label, color = Trainer.get_color_and_label(index, clinical_categories)
            colors.append(color)
            if label not in labels:
                patches_handles.append(
                    Patch(color=color, label=label)
                )
                labels.append(label)
        fig_size = CONFIG["FIG_SIZE"]
        other_arguments = dict()
        tick_font_size = fig_size[0] * 10 / len(feature_importance)
        if error is not None:
            other_arguments["xerr"] = error
            other_arguments["ecolor"] = "black"
            other_arguments["capsize"] = tick_font_size
        axes.barh(
            width=feature_importance,
            y=features,
            color=colors,
            **other_arguments
        )
        axes.margins(y=0)
        axes.tick_params(which="both")
        axes.tick_params(axis="y", labelsize=min(tick_font_size*2.7, CONFIG["TICK_SIZE"]))
        axes.legend(handles=patches_handles)
        axes.set_title(name, fontsize=CONFIG["TITLE_SIZE"])
        return axes

    @staticmethod
    def get_color_and_label(attribute: str, clinical_categories: Dict[str, str]) -> Tuple[str, str]:
        """
        Gets the category of the attribute given, and the
        color (in hex) assigned to it

        Parameters
        ----------
        attribute : str
            Name of the attribute
            In case of category this <attribute>_<category>
        clinical_categories : Dict[str, str]
            Attribute (or variable) with respective category

        Returns
        -------
        str
            Category of attribute
        str
            Color of the category
        """
        attribute_ = attribute
        if "_" in attribute:
            attribute_ = attribute.rsplit("_", 2)[0]
        category = clinical_categories[attribute_]
        color = CLINICAL_ATTRIBUTES[category]["color"]
        return category, color

    @staticmethod
    def __oversample__(data: pd.DataFrame, label: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Oversamples data to increase unbalanced class

        Parameters
        ----------
        data : DataFrame
            Training data to fit model
        label : Series
            Labels to predict with class

        Returns
        -------
        DataFrame
            Oversampled Training data
        Series
            Oversampled labels
        """
        logging.info("Oversampling data:")
        sampler = RandomOverSampler()
        return Trainer.__abstract_sample__(data, label, sampler)

    @staticmethod
    def __undersample__(data: pd.DataFrame, label: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Undersamples data to increase unbalanced class

        Parameters
        ----------
        data : DataFrame
            Training data to fit model
        label : Series
            Labels to predict with class

        Returns
        -------
        DataFrame
            Undersampled Training data
        Series
            Undersampled labels
        """
        logging.info("Undersampling data:")
        sampler = RandomUnderSampler()
        return Trainer.__abstract_sample__(data, label, sampler)

    @staticmethod
    def __abstract_sample__(data: pd.DataFrame,
                            label: pd.Series,
                            sampler: BaseSampler) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Abstract method for both method of resampling

        Parameters
        ----------
        data : DataFrame
            Training data to fit model
        label : Series
            Labels to predict with class
        sampler : BaseSampler
            Sampler to do resampling

        Returns
        -------
        DataFrame
            Resampled Training data
        Series
            Resampled labels
        """
        re_data, re_label = sampler.fit_resample(data, label)
        logging.info("\tOriginal set: {}\tResampled set: {}".format(
            len(data),
            len(re_data)
        ))
        re_data.index = data.index[sampler.sample_indices_]
        re_label.index = label.index[sampler.sample_indices_]
        return re_data, re_label
