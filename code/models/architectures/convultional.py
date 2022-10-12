""" convolutional architecture """

import math
import torch
import logging
from abc import ABC, abstractmethod
from models.architectures import ModifiedModule


class _PartialConvolutional(ModifiedModule, ABC):
    """
    Partial convolutional, without the final fully connected
    """
    @abstractmethod
    def __init__(self, device: torch.device) -> None:
        super(_PartialConvolutional, self).__init__(device)
        self.__conv__: ModifiedModule
        self.__register_module__("conv")
        return

    @staticmethod
    def calculate_padding_conv(sequence_length: int, kernel: int, stride: int) -> int:
        """
        Calculate padding of convolutional layer

        Parameters
        ----------
        sequence_length : int
            Original sequence length
        kernel : int
            Kernel size
        stride : int
            Stride size

        Returns
        -------
        int
            Padding to add
        """
        if stride != 1:
            remainder = (sequence_length - kernel) % stride
            total_padding = stride - remainder
            return math.ceil(total_padding / 2)
        else:
            return 0

    @staticmethod
    def calculate_out_conv(sequence_length: int, kernel: int, stride: int, padding: int) -> int:
        """
        Calculate output of convolutional layer

        Parameters
        ----------
        sequence_length : int
            Original sequence length
        kernel : int
            Kernel size
        stride : int
            Stride size
        padding : int
            Padding size

        Returns
        -------
        int
            Output of convolutional layer
        """
        return math.floor(((sequence_length + 2 * padding - kernel) / stride) + 1)

    @staticmethod
    def calculate_padding_pool(sequence_length: int, kernel: int) -> int:
        """
        Calculate padding of pool layer

        Parameters
        ----------
        sequence_length : int
            Original sequence length
        kernel : int
            Kernel size

        Returns
        -------
        int
            Padding
        """
        remainder = sequence_length % kernel
        total_padding = kernel - remainder
        return math.ceil(total_padding / 2)

    @staticmethod
    def calculate_out_pool(sequence_length: int, kernel: int, padding: int) -> int:
        """
        Calculate output size of pool layer

        Parameters
        ----------
        sequence_length : int
            Original sequence length
        kernel : int
            Kernel size
        padding : int
            Padding size

        Returns
        -------
        int
            Output size
        """
        return math.floor((sequence_length + 2 * padding) / kernel)

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
        return self.__conv__(x)


class _ConvolutionalFullyConnected(ModifiedModule, ABC):
    """
    Convolutional with a terminal fully connected
    """
    @abstractmethod
    def __init__(self, device: torch.device) -> None:
        super(_ConvolutionalFullyConnected, self).__init__(device)
        self.__conv__: ModifiedModule
        self.__fn__: ModifiedModule
        self.__register_module__("conv")
        self.__register_module__("fn")
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
        out = self.__conv__(x)
        return self.__fn__(out)


class ConvolutionalReduction(_PartialConvolutional):
    """
    Architecture to reduce data dimension (sequence length)
    """
    def __init__(self, sequence_length: int, channels: int, device: torch.device) -> None:
        super(ConvolutionalReduction, self).__init__(device)
        logging.info("Input length of sequence {}".format(sequence_length))
        kernel_size = [4, 20, 20]
        kernel_pool = [None, 20, None]
        kernel_stride = [1, 1, 10]
        padding = list()
        padding_pool = list()
        output_features = sequence_length
        for i in range(3):
            padding.append(self.calculate_padding_conv(output_features, kernel_size[i], kernel_stride[i]))
            output_features = self.calculate_out_conv(output_features, kernel_size[i], kernel_stride[i], padding[i])
            if kernel_pool[i] is not None:
                padding_pool.append(self.calculate_padding_pool(output_features, kernel_pool[i]))
                output_features = self.calculate_out_pool(output_features, kernel_pool[i], padding_pool[i])
            else:
                padding_pool.append(None)
        self.__input_fnn__ = output_features
        logging.info("Length of reduced sequence {}".format(output_features))
        self.__conv__ = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=(kernel_size[0], ),
                            padding=(padding[0], ), stride=(kernel_stride[0], )),
            torch.nn.Conv1d(channels, channels, kernel_size=(kernel_size[1], ),
                            padding=(padding[1], ), stride=(kernel_stride[1], )),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=(kernel_pool[1], ), padding=(padding_pool[1], )),
            torch.nn.Conv1d(channels, channels, kernel_size=(kernel_size[2], ),
                            padding=(padding[2], ), stride=(kernel_stride[2], )),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=(kernel_pool[2], ), padding=(padding_pool[2], )),
        ).float()
        return


class ConvolutionalFully(_ConvolutionalFullyConnected):
    """
    One Convolutional Layer architectures and a terminal fully connected
    """
    def __init__(self, sequence_length: int, channels: int, output_size: int, device: torch.device) -> None:
        super(ConvolutionalFully, self).__init__(device)
        self.__conv__ = ConvolutionalReduction(sequence_length, channels, device)
        self.__fn__ = torch.nn.Sequential(
            torch.nn.Linear(self.__conv__.__input_fnn__, output_size),
        )
        return


class Soybean(ModifiedModule):
    """
    Soybean architecture without final perceptron
    """
    def __init__(self, sequence_length: int,
                 channels: int, device: torch.device,
                 dropout_p: float = 0.75) -> None:
        super(Soybean, self).__init__(device)
        conv_kernel = (4, 20)
        conv_out_size = sequence_length - conv_kernel[0] + 1
        conv_out_size_2 = conv_out_size - conv_kernel[1] + 1
        self.__stacked__ = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                10,
                kernel_size=(conv_kernel[0], ),
            ),
            torch.nn.Conv1d(
                10,
                10,
                kernel_size=(conv_kernel[1], ),
            ),
            torch.nn.Dropout(p=dropout_p)
        ).float()
        self.__residual__ = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                10,
                kernel_size=(conv_kernel[0], ),
            ),
            torch.nn.Dropout(p=dropout_p)
        ).float()
        n_features = conv_out_size + conv_out_size_2
        conv_out_size_final = n_features - conv_kernel[0] + 1
        self.__input_fnn__ = conv_out_size_final * 10
        self.__feature_processing__ = torch.nn.Sequential(
            torch.nn.Conv1d(
                10,
                10,
                kernel_size=(conv_kernel[0], ),
            ),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=dropout_p),
        )
        self.__register_module__("stacked")
        self.__register_module__("residual")
        self.__register_module__("feature_processing")
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
        stacket_out = self.__stacked__(x)
        residual_out = self.__residual__(x)
        added = torch.cat((stacket_out, residual_out), dim=2)
        return self.__feature_processing__(added)


class SoybeanFully(_ConvolutionalFullyConnected):
    """
    Soybean architecture with final perceptron
    """
    def __init__(self, input_size: int, channels: int,
                 output_size: int, device: torch.device,
                 dropout_p: float = 0.75) -> None:
        super(SoybeanFully, self).__init__(device)
        self.__conv__ = Soybean(input_size, channels, device, dropout_p=dropout_p)
        self.__fn__ = torch.nn.Sequential(
            torch.nn.Linear(self.__conv__.__input_fnn__, output_size),
        )
