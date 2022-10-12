""" Checkpoint class """

import os
import pickle
import logging
from typing import List, Dict, Union


class Checkpoint:
    """
    Checkpoint class for picking up some training
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        self.__dataset__: List[str] = list()
        self.__model__: Dict[str, List[str]] = dict()
        self.__last__ = False
        self.__figures__: Dict[str, str] = dict()
        if os.path.exists("checkpoint.log"):
            self.__file__ = open("checkpoint.log", "r+")
            self.__read__()
        else:
            self.__file__ = open("checkpoint.log", "w")

    def __read__(self) -> None:
        """
        Reads log file to load checkpoint

        Return
        ----------
        None
        """
        logging.info("Reading checkpoint...")
        for line in self.__file__.readlines():
            formatted_line = line.strip("\n").split("\t")
            if len(formatted_line) == 2:
                dataset, model = formatted_line
                if dataset not in self.__dataset__:
                    self.__dataset__.append(dataset)
                    self.__model__[dataset] = list()
                self.__model__[dataset].append(model)
            elif len(formatted_line) == 0:
                pass
            else:
                raise RuntimeError("Wrong log formate file")
        if os.path.exists("checkpoint"):
            for fig in os.listdir("checkpoint"):
                self.__figures__[fig.split(".")[0]] = os.path.join("checkpoint", fig)
        return 

    def add_dataset(self, dataset: str) -> None:
        """
        Registers data file on checkpoint

        Parameters
        ----------
        dataset : str
            Name to identify dataset

        Returns
        -------
        None
        """
        self.__dataset__.append(dataset)
        self.__model__[dataset] = list()
        return

    def add_model(self, model: str) -> None:
        """
        Registers model on checkpoint

        Parameters
        ----------
        model : str
            Name to identify model

        Returns
        -------
        None
        """
        dataset = self.__dataset__[-1]
        if model not in self.__model__[dataset]:
            self.__model__[dataset].append(model)
            self.__file__.write(dataset + "\t" + model + "\n")
        return

    def last_dataset(self) -> None:
        """
        Prepares to delete file

        Returns
        -------
        None
        """
        self.__last__ = True
        return

    def last_model(self) -> None:
        """
        Delete log file if it is also last dataset

        Returns
        -------
        None
        """
        self.__figures__.clear()
        if os.path.exists("checkpoint"):
            for fig in os.listdir("checkpoint"):
                os.remove(os.path.join("checkpoint", fig))
            os.removedirs("checkpoint")
        if self.__last__:
            self.__dataset__.clear()
            self.__model__.clear()
            self.__file__.close()
            os.remove("checkpoint.log")
        return

    def has_dataset(self, dataset: str) -> bool:
        """
        Check if dataset is on checkpoint

        Parameters
        ----------
        dataset : str
            Dataset to check

        Returns
        -------
        bool
            If dataset is on checkpoint
        """
        is_in = dataset in self.__dataset__
        if is_in:
            logging.info("Dataset {} found...".format(dataset))
        return is_in

    def has_model(self, dataset: str, model: str) -> bool:
        """
        Check if model is on checkpoint

        Parameters
        ----------
        dataset : str
            Dataset to check
        model : str
            Model to check

        Returns
        -------
        bool
            If model is on checkpoint
        """
        is_in = False
        if dataset in self.__model__.keys():
            is_in = model in self.__model__[dataset]
        if is_in:
            logging.info("Model {} found, skipping training...".format(model))
        return is_in

    def save_object(self, obj: object, name: str) -> None:
        """
        Saves figure on disk as checkpoint

        Parameters
        ----------
        obj : object
            Object to save
        name : str
            Name to find object

        Returns
        -------
        None
        """
        os.makedirs("checkpoint", exist_ok=True)
        path = os.path.join("checkpoint", name + ".pkl")
        with open(path, "wb") as file:
            logging.info("Saving {} on {}".format(name, path))
            pickle.dump(obj, file)
            self.__figures__[name] = path
        return

    def get_object(self, name: str) -> Union[object, List[object]]:
        """
        Gets object associated with path

        Parameters
        ----------
        name : str
            Object name

        Returns
        -------
        object
            object saved or None in case there is none associated
        """
        if name in self.__figures__.keys():
            with open(self.__figures__[name], "rb") as file:
                return pickle.load(file)
        else:
            logging.warning("No figure associated with {}".format(name))
            return None
