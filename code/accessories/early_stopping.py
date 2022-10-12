""" early_stopping: Early Stopping class to stop training loop """

import logging


class EarlyStopping:
    """
    EarlyStopping class, adapted from

    https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """
        Constructor

        Parameters
        ----------
        patience : int
            Number of epoch without decreasing validation loss
        min_delta : float, optional
            Minimum delta of decreasing loss, default 0.0
        """
        self.__patience__ = patience
        self.__min_delta__ = min_delta
        self.__go_on__ = True
        self.__best_loss__ = float("inf")
        self.__historical_best__ = float("inf")
        self.__counter__ = 0
        self.__loss__ = float("inf")
        return

    def logging_loss(self) -> None:
        """
        Logs loss

        Returns
        -------
        None
        """
        if not self.__go_on__:
            logging.info("Early stopping with validation loss {:.2e}".format(self.__loss__))
        return

    def current_loss(self, loss: float) -> None:
        """
        Updates loss value to evaluate if the training continues

        Parameters
        ----------
        loss : float
            Current validation loss

        Returns
        -------
        None, updates object
        """
        self.__loss__ = loss
        if (self.__best_loss__ - loss) > self.__min_delta__:
            self.__best_loss__ = loss
            self.__counter__ = 0
            return
        elif (self.__best_loss__ - loss) <= self.__min_delta__:
            self.__counter__ += 1
            if self.__counter__ >= self.__patience__:
                self.__go_on__ = False
            return

    def go_on(self) -> bool:
        """
        Whether continue training

        Returns
        -------
        bool
            True if continue training
        """
        return self.__go_on__

    def reset(self) -> None:
        """
        Resets early stopping

        Returns
        -------
        None
        """
        self.__go_on__ = True
        self.__best_loss__ = float("inf")
        self.__loss__ = float("inf")
        self.__counter__ = 0
        return

    def best_historical(self) -> bool:
        """
        If best historical is significant more than current best

        Returns
        -------
        bool
            Whether current best is less than best historical
        """
        if (self.__historical_best__ - self.__loss__) > self.__min_delta__:
            self.__historical_best__ = self.__loss__
            return True
        else:
            return False
