""" Convolutional 1 dimension networks and forward fully connected neural networks """

from __future__ import annotations
import torch
import pandas as pd
from abc import ABC, abstractmethod
from models import NeuralNetwork
from models.architectures import SoybeanFully


class ConvFullyNeuralNetwork(NeuralNetwork, ABC):
    """
    Convolutional and Fully Connected
    Neural Network abstract class
    """
    @abstractmethod
    def __init__(self, name: str, sequence_length: int, channels: int, output_size: int,
                 learning_rate: float, weight: torch.Tensor = None, device_name: str = None) -> None:
        super(ConvFullyNeuralNetwork, self).__init__(
            name, learning_rate, weight=weight, device_name=device_name
        )
        self.__seq_length__ = sequence_length
        self.__channels__ = channels
        self.__output_size__ = output_size
        return

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
        return self.one_hot(torch.tensor(x_train.to_numpy()), self.__channels__)

    @staticmethod
    def one_hot(class_tensor: torch.tensor, classes: int) -> torch.tensor:
        """
        Return a copy of tensor in one hot encoding

        Parameters
        ----------
        class_tensor : tensor
            A tensor with dimension batch size x features
        classes : int
            Number of classes

        Returns
        -------
        tensor
            A tensor one hot encoded with dimensions
            batch size x classes x features
        """
        one_hot = torch.zeros(class_tensor.shape[0], classes, class_tensor.shape[1])
        one_hot.scatter_(1, class_tensor.unsqueeze(1).type("torch.LongTensor"), 1.0)
        del class_tensor
        return one_hot.float()

    def get_saliency(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Gets saliency values for each row and token in sequence.
        Calculated as:
        saliency = MAX(|dY/dX|)

        With Y the predicted value and X the input

        Parameters
        ----------
        x_test : DataFrame
            Data to calculate saliency

        Returns
        -------
        DataFrame
            Saliency values for each data in batch
        """
        x_test_tensor = self.as_tensor(x_test).requires_grad_()
        self.__model__.cpu()
        self.__model__.eval()
        output = self.__model__(x_test_tensor)
        output_idx = output.argmax(dim=1)
        output_max = output.gather(1, output_idx.view(-1, 1)).view(-1)
        output_max.backward(torch.ones_like(output_max, memory_format=torch.preserve_format))
        saliency, _ = torch.max(x_test_tensor.grad.data.abs(), dim=1)
        out = pd.DataFrame(
            saliency.detach().cpu().numpy(),
            index=x_test.index,
            columns=x_test.columns
        )
        return out


class Soybean(ConvFullyNeuralNetwork):
    """
    Soybean Architecture
    """
    def __init__(self, sequence_length: int, channels: int, output_size: int,
                 learning_rate: float, weight_decay: float = 1e-6, weight: torch.Tensor = None,
                 device_name: str = None, dropout_p: float = 0.75) -> None:
        super(Soybean, self).__init__(
            "Dual-stream CNN", sequence_length, channels,
            output_size, learning_rate, weight=weight, device_name=device_name
        )

        self.__model__ = SoybeanFully(
            sequence_length, channels,
            output_size, self.__device__, dropout_p=dropout_p
        ).float()
        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=self.__lr__, weight_decay=weight_decay)
        return
