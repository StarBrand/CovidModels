""" Recurrent network architectures """

import torch
from typing import Tuple
from models.architectures import ModifiedModule, ConvolutionalReduction


class RecurrentNN(ModifiedModule):
    """
    Convolutional to reduce dimensions and RNN
    """
    __LAYERS__ = 10
    __HIDDEN_STATES__ = 500

    def __init__(self, sequence_length: int,
                 channels: int, output_size: int,
                 device: torch.device, bidirectional: bool = False) -> None:
        super(RecurrentNN, self).__init__(device)
        self.__bi_dir__ = bidirectional
        self.__dim_red__ = ConvolutionalReduction(sequence_length, channels, self.__device__)
        self.__rnn__ = torch.nn.RNN(
            channels, RecurrentNN.__HIDDEN_STATES__, RecurrentNN.__LAYERS__,
            batch_first=True, bidirectional=self.__bi_dir__
        ).float()
        self.__fnn__ = torch.nn.Sequential(
            torch.nn.Linear(
                (2 ** self.__bi_dir__) * RecurrentNN.__HIDDEN_STATES__,
                300
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(300, output_size),
        ).float()
        self.__register_module__("dim_red")
        self.__register_module__("rnn")
        self.__register_module__("fnn")
        return

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Forward method of Module

        Parameters
        ----------
        x : tensor
            Input to calculate output

        Returns
        -------
        tensor
            Output
        """
        reduced = self.__dim_red__(x)
        hidden = self.init_hidden(reduced.shape[0])
        out, hidden = self.__rnn__(reduced.transpose(1, 2), hidden)
        return self.__fnn__(out[:, -1, :])

    def init_hidden(self, batch_size: int) -> torch.tensor:
        """
        Generate an empty hidden state

        Parameters
        ----------
        batch_size : int
            Batch size of current state

        Returns
        -------
        tensor
            Initial hidden state
        """
        layers = (2 ** self.__bi_dir__) * RecurrentNN.__LAYERS__
        return torch.zeros(
            layers,
            batch_size,
            RecurrentNN.__HIDDEN_STATES__
        ).float().to(self.__device__)


class LongShortTermMemory(RecurrentNN):
    """
    Convolutional to reduce dimensions and LSTM
    """
    def __init__(self, sequence_length: int,
                 channels: int, output_size: int,
                 device: torch.device, bidirectional: bool = False) -> None:
        super(LongShortTermMemory, self).__init__(
            sequence_length, channels, output_size,
            device, bidirectional=bidirectional
        )
        self.__rnn__ = torch.nn.LSTM(
            channels, RecurrentNN.__HIDDEN_STATES__, RecurrentNN.__LAYERS__,
            batch_first=True, bidirectional=self.__bi_dir__
        ).float()
        return

    def init_hidden(self, batch_size: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Generate an empty hidden state

        Parameters
        ----------
        batch_size : int
            Batch size of current state

        Returns
        -------
        tensor
            Initial hidden state
        tensor
            Initial cell state
        """
        return super(LongShortTermMemory, self).init_hidden(
            batch_size
        ), super(LongShortTermMemory, self).init_hidden(
            batch_size
        )
