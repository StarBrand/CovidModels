""" Deep Checkpoint for resuming Neural Networks training """

from typing import Generator
from accessories import Checkpoint


class DeepCheckpoint(Checkpoint):
    """
    DeepCheckpoint class
    Save the state on disk of ANN training
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        super().__init__()
        self.__active__ = False
        self.__iteration__: int = 0
        self.__current_fold__: int = 0
        self.__get_saved__()
        return

    def __get_saved__(self) -> None:
        """
        Gets saved information, about iterations and folds

        Returns
        -------
        None
        """
        saved_state = self.get_object("state")
        if saved_state is not None:
            self.__current_fold__ = saved_state[0]
            self.__iteration__ = saved_state[1]
        return

    def add_model(self, model: str) -> None:
        """
        Adds model to checkpoint

        Parameters
        ----------
        model : str
            Name of model

        Returns
        -------
        None
        """
        super(DeepCheckpoint, self).add_model(model)
        if self.__current_fold__ != 0:
            self.save_object(
                [self.__current_fold__, self.__iteration__],
                "state"
            )
        return

    def activate(self) -> None:
        """
        Activates checkpoint. This means, process is going to be resumed

        Returns
        -------
        None
        """
        self.__active__ = True
        return None

    def deactivate(self) -> None:
        """
        Deactivates checkpoint. This means, process is not going to be resumed
        and just save current status

        Returns
        -------
        None
        """
        self.__active__ = False
        return None

    def is_active(self) -> bool:
        """
        Whether checkpoint is active

        Returns
        -------
        bool
            Activation status
        """
        return self.__active__

    def save_indexes(self, k_folds: Generator) -> None:
        """
        Saves indexes of folds for CrossValidation

        Parameters
        ----------
        k_folds : Generator
            Generator of each fold

        Returns
        -------
        None
        """
        self.save_object(
            [(train_index, test_index) for (train_index, test_index) in k_folds],
            "k_folds"
        )
        return None

    def get_indexes(self) -> object:
        """
        Gets indexes of folds for CrossValidation

        Returns
        -------
        Generator or None
            Generator of train_index and test_index
        """
        return self.get_object("k_folds")

    def add_fold(self, fold: int) -> None:
        """
        Adds fold to checkpoint

        Parameters
        ----------
        fold : int
            Current fold

        Returns
        -------
        None
        """
        self.__current_fold__ = fold
        return

    def register_iter(self, iteration: int) -> None:
        """
        Registers current iteration

        Parameters
        ----------
        iteration : int
            Current iteration

        Returns
        -------
        None
        """
        self.__iteration__ = iteration
        return
