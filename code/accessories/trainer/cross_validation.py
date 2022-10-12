""" Cross Validation Wrapper """

from __future__ import annotations
import os
import gc
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve, auc
from accessories import Metric, DeepCheckpoint
from accessories.trainer import Trainer
from sklearn.model_selection import KFold, train_test_split
from accessories.config import CONFIG
from models import Model


class CrossValidation(Trainer):
    """
    Cross Validation, trains a model by using
    cross validation method

    Attributes
    ----------
    __fpr__ : List[NDArray]
        False positive rates, one per fold
    __tpr__ : List[NDArray]
        True positive rates, one per fold
    __auc__ : List[float]
        Area Under the Curve, one per fold
    """
    def __init__(self, multiclass: bool = False, labels: List[str] = None) -> None:
        """
        Constructor
        """
        super(CrossValidation, self).__init__(multiclass=multiclass, labels=labels)
        self.__fpr__: List[np.ndarray] = list()
        self.__tpr__: List[np.ndarray] = list()
        self.__auc__: List[float] = list()
        self.__fold_metric__: Dict[int, Metric] = dict()
        return

    def train_model(self, model: Model, x: pd.DataFrame, y: pd.Series, resampling: str = None, **kwargs) -> None:
        """
        Trains model using cross validation

        Parameters
        ----------
        model : Model
            Model to be trained by cross validation
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
        no_training_data : bool, optional
            Not call training data (for ML models)
        plot_validation : bool, optional
            If plot testing
        saliency : str, optional
            If given path to save saliency map
        attention : str, optional
            If given path to save attention
        folds : int
            Number of folds to use, default k=5
        fold_metric : bool
            Calculate metric for each folds
        checkpoint : DeepCheckpoint
            DeepCheckpoint object for ANN training
        kwargs
            Other parameters of model.train method

        Returns
        -------
        None
        """
        self.__fpr__.clear()
        self.__tpr__.clear()
        self.__auc__.clear()
        folds = kwargs.get("folds", CONFIG["K_FOLD"])
        k_fold = KFold(n_splits=folds, shuffle=True)
        y_true = pd.Series(dtype="float64")
        y_pred = pd.Series(dtype="float64")
        self.__training_data__.clear()
        fold = 0
        val_set = kwargs.get("plot_validation", False)
        path_to_model = None
        if "path_to_model" in kwargs.keys():
            path_to_model = os.path.join(
                kwargs.get("path_to_model"),
                model.__name__
            )
            kwargs.pop("path_to_model")
        saliency_data = pd.DataFrame(dtype="float64")
        attention_data_pre_trained = pd.DataFrame(dtype="float64")
        attention_data_fine_tuned = pd.DataFrame(dtype="float64")
        fold_generator = k_fold.split(x)
        checkpoint: DeepCheckpoint = None
        if "checkpoint" in kwargs.keys():
            checkpoint = kwargs.get("checkpoint")
            pred_true = checkpoint.get_object("y")
            if pred_true is not None:
                y_pred = pred_true[0]
                y_true = pred_true[1]
            _auc_ = checkpoint.get_object("auc")
            fpr = checkpoint.get_object("fpr")
            tpr = checkpoint.get_object("tpr")
            if _auc_ is not None:
                self.__auc__ = _auc_
            if fpr is not None:
                self.__fpr__ = fpr
            if tpr is not None:
                self.__tpr__ = tpr
            if not kwargs.get("no_training_data", False):
                train_data = checkpoint.get_object("train_data")
                if train_data is not None:
                    self.__training_data__ = train_data
            if "saliency" in kwargs.keys():
                saliency_data = checkpoint.get_object("saliency")
                if saliency_data is None:
                    saliency_data = pd.DataFrame(dtype="float64")
            if checkpoint.is_active():
                fold_generator = checkpoint.get_indexes()
            else:
                checkpoint.save_indexes(k_fold.split(x))
        for train_index, test_index in fold_generator:
            fold += 1
            logging.info("\tFold {:02d}/{:02d}".format(fold, kwargs.get("folds", CONFIG["K_FOLD"])))
            if checkpoint is not None:
                if checkpoint.__current_fold__ > fold:
                    logging.info("\tFold done, skipping")
                    continue
                elif checkpoint.__current_fold__ == fold:
                    logging.info("\tCurrent fold")
                    checkpoint.activate()
                else:
                    logging.info("\tNew fold, saving")
                    checkpoint.add_fold(fold)
                    checkpoint.deactivate()
            if "attention" in kwargs.keys():
                model.__model__.cpu()
                try:
                    if "attention_batch_size" in kwargs.keys():
                        attention_data_pre_trained = pd.concat((
                            attention_data_pre_trained, model.get_attention(
                                x.iloc[test_index, ], batch_size=kwargs["attention_batch_size"]
                            )
                        ))
                    else:
                        attention_data_pre_trained = pd.concat((
                            attention_data_pre_trained, model.get_attention(x.iloc[test_index, ])
                        ))
                except AttributeError:
                    logging.warning("Model doesn't calculate attention")
                model.__model__.to(model.__device__)
            if val_set:
                x_train, x_val, y_train, y_val = train_test_split(
                    x.iloc[train_index, :], y.iloc[train_index],
                    test_size=1/(folds - 1)
                )
                if resampling is not None:
                    if resampling == "under":
                        x_train, y_train = self.__undersample__(x_train, y_train)
                    elif resampling == "over":
                        x_train, y_train = self.__oversample__(x_train, y_train)
                    else:
                        raise AttributeError("Sampling method {} not valid".format(resampling))
                if path_to_model is None:
                    trained_model = model.train(x_train, y_train, val_set=(x_val, y_val), **kwargs)
                else:
                    trained_model = model.train(
                        x_train, y_train, val_set=(x_val, y_val),
                        path_to_model=path_to_model + "{:02d}fold".format(fold), **kwargs
                    )
            else:
                x_train = x.iloc[train_index, :]
                y_train = y.iloc[train_index]
                if resampling is not None:
                    if resampling == "under":
                        x_train, y_train = self.__undersample__(x_train, y_train)
                    elif resampling == "over":
                        x_train, y_train = self.__oversample__(x_train, y_train)
                    else:
                        raise AttributeError("Sampling method {} not valid".format(resampling))
                trained_model = model.train(x_train, y_train, **kwargs)
            if path_to_model is not None:
                trained_model.save_model(path_to_model + "{:02d}fold".format(fold))
            probabilities = trained_model.classify(x.iloc[test_index, ], **kwargs)
            prediction = trained_model.predict(x.iloc[test_index, ], **kwargs)
            if "saliency" in kwargs.keys():
                try:
                    saliency_data = pd.concat((
                        saliency_data, trained_model.get_saliency(x.iloc[test_index, ])
                    ))
                    if checkpoint is not None:
                        checkpoint.save_object(saliency_data, "saliency")
                except AttributeError:
                    logging.warning("Model doesn't calculate saliency")
            if "attention" in kwargs.keys():
                try:
                    if "attention_batch_size" in kwargs.keys():
                        attention_data_pre_trained = pd.concat((
                            attention_data_pre_trained, model.get_attention(
                                x.iloc[test_index, ], batch_size=kwargs["attention_batch_size"]
                            )
                        ))
                    else:
                        attention_data_pre_trained = pd.concat((
                            attention_data_pre_trained, model.get_attention(x.iloc[test_index, ])
                        ))
                except AttributeError:
                    logging.warning("Model doesn't calculate attention")
            if kwargs.get("fold_metric", False):
                self.__fold_metric__[fold] = Metric(
                    model.__name__,
                    prediction,
                    y.iloc[test_index],
                    multiclass=self.__multi__
                )
            y_pred = pd.concat([y_pred, prediction])
            y_true = pd.concat([y_true, y.iloc[test_index]])
            __fpr__ = np.linspace(0, 1)
            if self.__multi__:
                true_prob = pd.Series(dtype="float64")
                for j in probabilities.columns:
                    true_prob = pd.concat([true_prob, (y.iloc[test_index] == j) * 1.0])
                fpr, tpr, _ = roc_curve(true_prob, probabilities.to_numpy().ravel())
            else:
                fpr, tpr, _ = roc_curve(y.iloc[test_index], probabilities[1])
            __tpr__ = np.interp(__fpr__, fpr, tpr)
            __tpr__[0] = 0.0
            self.__auc__.append(auc(fpr, tpr))
            self.__fpr__.append(__fpr__)
            self.__tpr__.append(__tpr__)
            if checkpoint is not None:
                checkpoint.save_object([y_pred, y_true], "y")
                checkpoint.save_object(self.__auc__, "auc")
                checkpoint.save_object(self.__fpr__, "fpr")
                checkpoint.save_object(self.__tpr__, "tpr")
            if not kwargs.get("no_training_data", False):
                self.__training_data__[fold] = trained_model.train_data()
                if checkpoint is not None:
                    checkpoint.save_object(self.__training_data__, "train_data")
            del trained_model
            gc.collect()
        if "saliency" in kwargs.keys():
            logging.info("Saving saliency data on {}".format(kwargs.get("saliency")))
            saliency_data.median(axis=0).to_csv(kwargs.get("saliency"), sep=",", index=True, header=False)
        if "attention" in kwargs.keys():
            attention_folder = kwargs.get("attention")
            logging.info("Saving attention data on {}".format(attention_folder))
            attention_data_pre_trained.to_csv(
                os.path.join(attention_folder, model.__name__ + "_pre_trained.csv"),
                sep=",", index=True, header=False)
            attention_data_fine_tuned.to_csv(
                os.path.join(attention_folder, model.__name__ + "_fine_tuned.csv"),
                sep=",", index=True, header=False)
        self.__metric__ = Metric(model.__name__, y_pred, y_true, multiclass=self.__multi__)
        return

    def plot_roc_curve(self, axes: Axes) -> Axes:
        """
        Plots Receiver Operating Characteristic (ROC) curve
        in Cross Validation Context

        Adapted from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

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
        for fold in range(len(self.__fpr__)):
            axes.plot(
                self.__fpr__[fold], self.__tpr__[fold], alpha=0.3, lw=1,
                label="{}ROC fold {} (AUC = {:.2f})".format(disclaimer, fold + 1, self.__auc__[fold])
            )
        axes.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(np.array(self.__tpr__), axis=0)
        mean_fpr = np.mean(np.array(self.__fpr__), axis=0)
        mean_auc = np.mean(self.__auc__)
        std_aux = np.std(self.__auc__)
        axes.plot(
            mean_fpr, mean_tpr, color="b", lw=2, alpha=0.8,
            label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_aux)
        )
        std_tpr = np.std(np.array(self.__tpr__), axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        axes.fill_between(
            mean_fpr, tpr_lower, tpr_upper,
            color="grey", alpha=0.2,
            label=r'$\pm$ $\sigma$'
        )
        return super(CrossValidation, self).plot_roc_curve(axes)

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
        metric_val_axes : Axes, optional
            matplotlib.axes.Axes where to plot testing metric
        loss_val_axes : Axes, optional
            matplotlib.axes.Axes where to plot testing loss

        Other Parameters
        ----------------
        None

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
        for fold in self.__training_data__:
            metric_axes.plot(
                self.__training_data__[fold]["Learning metric"],
                label="Fold={}".format(fold)
            )
            loss_axes.plot(
                self.__training_data__[fold]["Loss"],
                "--", label="Fold={}".format(fold))
            metric_val_axes.plot(
                self.__training_data__[fold]["Testing metric"],
                label="Fold={}".format(fold)
            )
            p = loss_val_axes.plot(
                self.__training_data__[fold]["Test Loss"],
                "--", label="Fold={}".format(fold))
            if len(self.__training_data__[fold]["Early stops"]) > 0:
                for epoch, loss in self.__training_data__[fold]["Early stops"]:
                    loss_val_axes.annotate("", xy=(epoch, loss), xycoords="data",
                                           xytext=(0, 25), textcoords="offset points",
                                           arrowprops=dict(facecolor=p[-1].get_color(), shrink=0.05))
        return super(CrossValidation, self)._postprocessing_training(
            (metric_axes, loss_axes, metric_val_axes, loss_val_axes), legend=True
        )

    def get_metric_per_fold(self, metric_function: str, report: pd.DataFrame = None) -> pd.DataFrame:
        """
        Gets a summary metric for each fold

        Parameters
        ----------
        metric_function : str
            Name of metric, valid:
            accuracy, precision, recall, f1-score
        report : DataFrame
            Report to add metrics

        Returns
        -------
        DataFrame
            Metric per fold, complete and standard error
        """
        def get_metric(metric: Metric):
            """
            Calculates metric given

            Parameters
            ----------
            metric : Metric
                Metric object to get metric

            Returns
            -------
            float
                Value of metric
            """
            if metric_function == "accuracy":
                return metric.accuracy()
            elif metric_function == "precision":
                return metric.precision()
            elif metric_function == "recall":
                return metric.recall()
            elif metric_function == "f1-score":
                return metric.f1_score()

        output = dict()
        output[metric_function] = get_metric(self.__metric__)
        all_metrics = list()
        for fold in self.__fold_metric__:
            output[fold] = get_metric(self.__fold_metric__[fold])
            all_metrics.append(get_metric(self.__fold_metric__[fold]))
        output["sd"] = np.std(all_metrics, ddof=1)
        output["se"] = output["sd"] / np.sqrt(len(all_metrics))
        output = pd.DataFrame(output, index=[self.__metric__.__name__])
        if report is not None:
            return pd.concat((report, output))
        else:
            return output

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
        if "Feature Importance" not in self.__training_data__[1].keys():
            raise AttributeError("Model not support feature importance")
        feature_importance = pd.DataFrame(dtype=float)
        threshold = 1e-5
        same_threshold = True
        for fold in self.__training_data__.keys():
            temp = pd.DataFrame(
                self.__training_data__[fold]["Feature Importance"],
                columns=[fold], index=self.__training_data__[fold]["Features"]
            )
            if self.__training_data__[fold]["Threshold"] != threshold:
                same_threshold = False
            if len(feature_importance) == 0:
                feature_importance = temp
            else:
                feature_importance = pd.concat([
                    feature_importance, temp
                ], axis=1)
            del temp
        feature_importance["Feature Importance"] = feature_importance.mean(axis=1)
        feature_importance["Standard Deviation"] = feature_importance.std(axis=1)
        if not same_threshold:
            threshold = feature_importance["Feature Importance"].mean()
        mask = (feature_importance["Feature Importance"] - feature_importance["Standard Deviation"]) > threshold
        feature_importance = feature_importance.loc[
            feature_importance[mask].index,
            ["Feature Importance", "Standard Deviation"]
        ]
        feature_importance.sort_values("Feature Importance", ascending=True, inplace=True)
        if feature_selection_path is not None:
            feature_importance["Variable"] = feature_importance.index.str.split("_").str[0]
            feature_importance[["Variable", "Feature Importance", "Standard Deviation"]].to_csv(
                feature_selection_path,
                header=True, index=True, sep=","
            )
        return super(CrossValidation, self).__plot_feature_selection__(
            self.__metric__.__name__ + "\n", axes,
            feature_importance["Feature Importance"],
            feature_importance["Standard Deviation"]
        )
