""" Neural Network Abstract Class """

import os
import gc
import torch
import math
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Tuple
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from accessories import EarlyStopping, DeepCheckpoint, show_percentage_of_process
from accessories.config import CONFIG
from models import Model

OPTIMIZER_OPTIONS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}


class NeuralNetwork(Model, ABC):
    """
    Neural Network Abstract Class
    """
    @abstractmethod
    def __init__(self, name: str,
                 learning_rate: float,
                 op_algorithm: str = "adam",
                 weight: torch.Tensor = None,
                 device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        name : str
            Name for reports
        learning_rate : float
            Learning rate of NN
        op_algorithm : str, optional
            Optimization algorithm, by default adam
        weight: torch.Tensor, optional
            Rescaling weight for each class
        device_name : str, optional
            Device to run tensors in
        """
        super().__init__(name)
        self.__model__ = torch.nn.Module()  # Place Holder
        self.__loss_fn__ = torch.nn.CrossEntropyLoss(weight=weight)
        self.__val_loss__ = torch.nn.CrossEntropyLoss(weight=weight)
        self.__lr__ = learning_rate
        try:
            self.__optimizer_callable__ = OPTIMIZER_OPTIONS[op_algorithm]
        except KeyError:
            logging.error("{} not implemented optimization algorithm".format(op_algorithm))
            logging.error("Available algorithms {}".format(", ".join(list(OPTIMIZER_OPTIONS.keys()))))
            raise NotImplementedError("Not implemented optimizer {}".format(op_algorithm))
        self.__optimizer__ = None
        self.__losses__ = list()
        self.__learn_curve__ = list()
        self.__val_losses__ = list()
        self.__val_curve__ = list()
        self.__early_stops__: List[Tuple[int, float]] = list()
        if device_name is None:
            self.__device__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.__device__ = torch.device(device_name if torch.cuda.is_available() else "cpu")
        logging.info("{} computing using {}{}".format(
            self.__name__,
            self.__device__.type,
            ":{}".format(self.__device__.index) if self.__device__.index is not None else ""
        ))
        return

    def train(self, x_train: pd.DataFrame, y_train: Union[np.ndarray, pd.Series], **kwargs) -> Model:
        """
        Train and return a trained model

        Parameters
        ----------
        x_train : DataFrame
            Parameters to train model
        y_train : ND-array or Series
            Labels to classify

        Other Parameters
        ----------------
        val_set : DataFrame, ND-array or Series
            [Mandatory] To calculate loss and metric of validation set
        batch_size : int, optional
            Size of batch, if not given use whole train set
        epoch : int
            Epoch to do training
        early_stopping : bool
            If the train stop when validation loss stop decreasing, default False
            If val_set is not given, early_stopping is assumed to be False, even if it is given as True
        patience : int
            Patience of early stop. If none given, early_stopping is assumed to be False
        min_delta : float
            Minimum delta of early stopping. If non given, its value is the default one (0.0)
        path_to_model : str, optional
            If given, saves the model to path
        writer : SummaryWriter
            TensorBoard logging
        checkpoint : DeepCheckpoint
            DeepCheckpoin object for resuming training

        Returns
        -------
        Model
            Trained model
        """

        def get_metric(actual: torch.tensor, expected: torch.tensor) -> float:
            """
            Gets metric calculates from actual and expected values

            Parameters
            ----------
            actual : tensor
                Values obtained from trained model
            expected : tensor
                Label given for training

            Returns
            -------
            float
                Value of metric (f1-score)
            """
            try:
                return f1_score(torch.argmax(actual, dim=1).detach().cpu().numpy(),
                                expected, average="binary", zero_division=0)
            except ValueError:
                return f1_score(torch.argmax(actual, dim=1).detach().cpu().numpy(),
                                expected, average="micro", zero_division=0)

        writer: SummaryWriter = None
        if "writer" in kwargs.keys():
            writer = kwargs.get("writer")
        checkpoint: DeepCheckpoint = None
        if "checkpoint" in kwargs.keys():
            checkpoint = kwargs.get("checkpoint")

        model = deepcopy(self)
        model.__model__.to(model.__device__)
        loss_fn = model.__loss_fn__.to(model.__device__)

        train_size = x_train.shape[0]
        batch_size = kwargs.get("batch_size", None)
        if batch_size is None:
            batch_size = x_train.shape[0]
        logging.info("Using batch size of {}".format(batch_size))

        # Validation set
        if "val_set" not in kwargs.keys():
            raise RuntimeError("Training networks must have validation test")
        x_val = kwargs.get("val_set")[0]
        y_val = kwargs.get("val_set")[1]
        early_stopping = kwargs.get("early_stopping", False)
        check_early = None
        if early_stopping and "patience" in kwargs.keys():
            if "min_delta" in kwargs.keys():
                check_early = EarlyStopping(kwargs.get("patience"), kwargs.get("min_delta"))
            else:
                check_early = EarlyStopping(kwargs.get("patience"))

        epochs = kwargs.get("epochs", CONFIG["EPOCHS"])

        init_epoch = 0
        checkpoint_path = os.path.join(
            os.path.dirname(kwargs.get("path_to_model")),
            model.__name__ + "_checkpoint"
        )
        if checkpoint is not None and checkpoint.is_active():
            if checkpoint.__iteration__ != 0:
                init_epoch = checkpoint.__iteration__ + 1
                logging.info("Resuming training from epoch {}".format(init_epoch))
                model.load_model(checkpoint_path)
                _losses_ = checkpoint.get_object("losses")
                _learn_ = checkpoint.get_object("learn")
                _val_ = checkpoint.get_object("val")
                _val_losses_ = checkpoint.get_object("val_losses")
                if _losses_ is not None:
                    model.__losses__ = _losses_
                if _learn_ is not None:
                    model.__learn_curve__ = _learn_
                if _val_ is not None:
                    model.__val_curve__ = _val_
                if _val_losses_ is not None:
                    model.__val_losses__ = _val_losses_

        for epoch in range(init_epoch, epochs):
            if early_stopping and not check_early.go_on():
                if check_early.best_historical():
                    check_early.logging_loss()
                    model.save_model(kwargs.get("path_to_model"))
                    model.__early_stops__.append((epoch, check_early.__loss__))
                check_early.reset()

            model.__model__.train()
            i = 0
            j = batch_size
            iterations = math.ceil(train_size / batch_size)
            out = torch.tensor([])
            batch_y = torch.tensor([])
            loss_train = float("nan")
            for _ in range(iterations):
                del out
                if batch_size != x_train.shape[0]:
                    print(show_percentage_of_process(train_size, j), end="\r")
                batch_x = x_train.iloc[i:j, :]
                batch_y = y_train.iloc[i:j]

                out = model(batch_x)
                batch_y_on_gpu = torch.tensor(batch_y.to_numpy()).to(model.__device__)
                loss = loss_fn(out, batch_y_on_gpu)
                del batch_y_on_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                model.__optimizer__.zero_grad()
                loss.backward()
                model.__optimizer__.step()
                loss_train = loss.item()

                i = j
                j += batch_size
                if j >= train_size:
                    j = train_size

            model.__losses__.append(loss_train)
            metric = get_metric(out, batch_y)
            model.__learn_curve__.append(metric)
            if writer is not None:
                writer.add_scalar("F1-score/train", metric, epoch)
                writer.add_scalar("Loss/train", loss_train, epoch)

            with torch.no_grad():
                model.__model__.eval()
                if batch_size == train_size:
                    val_out = model(x_val).to(model.__device__)
                else:
                    val_out = model.forward_batches(x_val, batch_size, cpu=False).to(model.__device__)
                loss_val = model.__val_loss__(val_out.cpu(), torch.tensor(y_val.to_numpy()))
                model.__val_losses__.append(loss_val.item())
                val_metric = get_metric(val_out, y_val)
                model.__val_curve__.append(val_metric)
                if early_stopping:
                    check_early.current_loss(loss_val.item())
                if writer is not None:
                    writer.add_scalar("F1-score/validation", val_metric, epoch)
                    writer.add_scalar("Loss/validation", loss_val.item(), epoch)

            if (epoch + 1) % int(epochs // 10) == 0:
                logging.info("\tEpoch {:d}, Training loss {:.4f}, Validation loss {:.4f}".format(
                    epoch + 1,
                    loss_train,
                    loss_val.item()
                ))
                if checkpoint is not None:
                    model.save_model(checkpoint_path)
                    checkpoint.register_iter(epoch)
                    checkpoint.save_object(model.__losses__, "losses")
                    checkpoint.save_object(model.__learn_curve__, "learn")
                    checkpoint.save_object(model.__val_curve__, "val")
                    checkpoint.save_object(model.__val_losses__, "val_losses")

            del val_out, loss_val

        if checkpoint is not None:
            checkpoint.register_iter(0)

        if early_stopping:
            if not check_early.best_historical():
                logging.info("Early stopping checkpoint")
                model.load_model(kwargs.get("path_to_model"))

        del loss_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        model.__model__.cpu()
        return model

    def as_tensor(self, x_train: pd.DataFrame) -> torch.tensor:
        """
        Convert train set from dataframe to tensor

        Parameters
        ----------
        x_train : DataFrame
            Train set

        Returns
        -------
        tensor
            Train set as tensor
        """
        return torch.tensor(x_train.to_numpy()).double()

    def __call__(self, *args, **kwargs) -> torch.tensor:
        """
        Wrapper of torch.nn.Module.__call__

        Parameters
        ----------
        args[0] : DataFrame
            Input train
        cpu : bool
            Calculate on cpu and detached

        Returns
        -------
        tensor
            Output
        """
        if kwargs.get("cpu", False):
            x_input = self.as_tensor(args[0]).detach().cpu()
        else:
            x_input = self.as_tensor(args[0]).to(self.__device__)
        out = self.__model__(x_input)
        del x_input
        return out

    def predict(self, x_test: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Predicts test data and return the prediction

        Parameters
        ----------
        x_test : DataFrame
            The test data

        Other Parameters
        ----------------
        batch_size : int
            Batch size, to fit memory
        cpu : bool
            Calculate on cpu, default True

        Returns
        -------
        Series
            Predicted Labels
        """
        batch_size = kwargs.get("batch_size", None)
        if batch_size is None:
            y_pred = self(x_test, cpu=kwargs.get("cpu", True))
        else:
            y_pred = self.forward_batches(x_test, batch_size, kwargs.get("cpu", True))
        pred = torch.argmax(y_pred, dim=1)
        return pd.Series(pred.detach().cpu().numpy(), index=x_test.index)

    def classify(self, x_test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Use the model to predict

        Parameters
        ----------
        x_test : DataFrame
            Input data

        Other Parameters
        ----------------
        batch_size : int
            Batch size, to fit memory
        cpu : bool
            Calculate on cpu

        Returns
        -------
        DataFrame:
            Probabilities
        """
        batch_size = kwargs.get("batch_size", None)
        if batch_size is None:
            y_pred = self(x_test, cpu=kwargs.get("cpu", True))
        else:
            y_pred = self.forward_batches(x_test, batch_size, kwargs.get("cpu", True))
        y_pred = torch.nn.Softmax(dim=1)(y_pred)
        return pd.DataFrame(y_pred.detach().cpu().numpy(), index=x_test.index)

    def forward_batches(self, x_input: pd.DataFrame, batch_size: int, cpu: bool = False) -> torch.tensor:
        """
        Calculate output in batches

        Parameters
        ----------
        x_input : DataFrame
            Input from calculation
        batch_size : int
            Size of batch
        cpu : bool
            Calculate on cpu, default False

        Returns
        -------
        tensor
            Output of forward (model())
        """
        batched_dim = x_input.shape[0]
        prediction = None
        i = 0
        j = batch_size
        iterations = math.ceil(batched_dim / batch_size)
        for _ in range(iterations):
            print(show_percentage_of_process(batched_dim, j), end="\r")
            on_calculation = x_input.iloc[i:j, :]
            if prediction is None:
                prediction = self(on_calculation, cpu=cpu)
            else:
                prediction = torch.cat(
                    (prediction, self(on_calculation, cpu=cpu)),
                    dim=0
                )
            del on_calculation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            i = j
            j += batch_size
            if j >= batched_dim:
                j = batched_dim
        return prediction

    def train_data(self) -> Dict[str, Any]:
        """
        Gets Data generated during training

        Returns
        -------
        Dict
            Dictionary with data
        """
        return {
            "Loss": self.__losses__.copy(),
            "Learning metric": self.__learn_curve__.copy(),
            "Testing metric": self.__val_curve__.copy(),
            "Test Loss": self.__val_losses__.copy(),
            "Early stops": self.__early_stops__.copy(),
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
        name = os.path.basename(path)
        logging.info("Saving Model {} to {}".format(name, os.path.dirname(path)))
        torch.save(self.__model__.state_dict(), path)
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
        name = os.path.basename(path)
        logging.info("Loading Model {} from {}".format(self.__name__, path))
        self.__model__.load_state_dict(torch.load(path))
        self.__model__.eval()
        return
