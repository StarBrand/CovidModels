"""
Recurrent Neural Networks with dimensional reduction
with CNN and forward fully connected neural networks
"""

from __future__ import annotations
import torch
from models import ConvFullyNeuralNetwork
from models.architectures import RecurrentNN, LongShortTermMemory


class RNN(ConvFullyNeuralNetwork):
    """
    Convolutional reduction, Recurrent Neural Network and fully connected
    """
    def __init__(self, sequence_length: int, channels: int,
                 output_size: int, learning_rate: float,
                 weight: torch.Tensor = None, device_name: str = None) -> None:
        super(RNN, self).__init__(
            "RNN", sequence_length, channels, output_size, learning_rate, weight=weight, device_name=device_name
        )
        self.__model__ = RecurrentNN(
            sequence_length, channels, output_size, self.__device__
        )
        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=self.__lr__)
        return


class BidirectionalRNN(ConvFullyNeuralNetwork):
    """
    Convolutional reduction, Bidirectional Recurrent Neural Network and fully connected
    """
    def __init__(self, sequence_length: int, channels: int, output_size: int,
                 learning_rate: float, device_name: str = None) -> None:
        super(BidirectionalRNN, self).__init__(
            "Bidirectional RNN", sequence_length, channels, output_size, learning_rate, device_name=device_name
        )
        self.__model__ = RecurrentNN(
            sequence_length, channels, output_size, self.__device__, bidirectional=True
        )
        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=self.__lr__)
        return


class LSTM(ConvFullyNeuralNetwork):
    """
    Convolutional reduction, Long-Short Term Memory and fully connected
    """
    def __init__(self, sequence_length: int, channels: int, output_size: int,
                 learning_rate: float, device_name: str = None) -> None:
        super(LSTM, self).__init__(
            "LSTM", sequence_length, channels, output_size, learning_rate, device_name=device_name
        )
        self.__model__ = LongShortTermMemory(
            sequence_length, channels, output_size, self.__device__
        )
        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=self.__lr__)
        return


class BidirectionalLSTM(ConvFullyNeuralNetwork):
    """
    Convolutional reduction, Bidirectional Long-Short Term Memory and fully connected
    """
    def __init__(self, sequence_length: int, channels: int, output_size: int,
                 learning_rate: float, device_name: str = None) -> None:
        super(BidirectionalLSTM, self).__init__(
            "Bidirectional LSTM", sequence_length, channels, output_size,
            learning_rate, device_name=device_name
        )
        self.__model__ = LongShortTermMemory(
            sequence_length, channels, output_size, self.__device__
        )
        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=self.__lr__)
        return
