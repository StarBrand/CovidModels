""" Fully Connected Neural Networks Models """

from abc import ABC, abstractmethod
from models import NeuralNetwork
from models.architectures import FullTwoLayers, FullThreeLayers, FullFiveLayers, FullNineLayers


class FNN(NeuralNetwork, ABC):
    """
    Fully Connected standard methods
    """
    @abstractmethod
    def __init__(self, name: str, learning_rate: float, op_algorithm: str = "adam", device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        name : str
            Name for reports
        learning_rate : float
            Learning rate of NN
        device_name : str, optional
            Device to run tensors in
        """
        super(FNN, self).__init__(name, learning_rate, op_algorithm=op_algorithm, device_name=device_name)
        return


class TwoLayers(FNN):
    """
    Two Layers model (Input layer + ReLu, Output)
    """

    def __init__(self, input_size: int, output_size: int, learning_rate: float,
                 op_algorithm: str = "adam", device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        input_size : int
            Size of input
        output_size : int
            Size of output
        learning_rate : float
            Learning rate of model
        op_algorithm : str, optional
            Optimization algorithm, by default adam
        device_name : str, optional
            Device to run tensors in
        """
        super(TwoLayers, self).__init__("Two Layers NN", learning_rate,
                                        op_algorithm=op_algorithm, device_name=device_name)
        self.__model__ = FullTwoLayers(input_size, output_size, self.__device__).double()
        self.__optimizer__ = self.__optimizer_callable__(self.__model__.parameters(), lr=self.__lr__)
        return


class ThreeLayers(FNN):
    """
    Three Layers model (Input layer + ReLu, Hidden Layer + Tanh, Output + softmax)
    """

    def __init__(self, input_size: int, output_size: int, learning_rate: float,
                 op_algorithm: str = "adam", device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        input_size : int
            Size of input
        output_size : int
            Size of output
        learning_rate : float
            Learning rate of model
        op_algorithm : str, optional
            Optimization algorithm, by default adam
        device_name : str, optional
            Device to run tensors in
        """
        super(ThreeLayers, self).__init__("Three Layers NN", learning_rate,
                                          op_algorithm=op_algorithm, device_name=device_name)
        self.__model__ = FullThreeLayers(input_size, output_size, self.__device__).double()
        self.__optimizer__ = self.__optimizer_callable__(self.__model__.parameters(), lr=self.__lr__)
        return


class FiveLayers(FNN):
    """
    Three Layers model (Input layer + Relu, Hidden Layer * 4 + Relu, Output + softmax)
    """

    def __init__(self, input_size: int, output_size: int, learning_rate: float,
                 op_algorithm: str = "adam", device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        input_size : int
            Size of input
        output_size : int
            Size of output
        learning_rate : float
            Learning rate of model
        op_algorithm : str, optional
            Optimization algorithm, by default adam
        device_name : str, optional
            Device to run tensors in
        """
        super(FiveLayers, self).__init__("Five Layers NN", learning_rate,
                                         op_algorithm=op_algorithm, device_name=device_name)
        self.__model__ = FullFiveLayers(input_size, output_size, self.__device__).double()
        self.__optimizer__ = self.__optimizer_callable__(self.__model__.parameters(), lr=self.__lr__)
        return


class NineLayers(FNN):
    """
    Nine Layers model (Input layer + Relu, Hidden Layer + Relu * 8, Output + softmax)
    """

    def __init__(self, input_size: int, output_size: int, learning_rate: float,
                 op_algorithm: str = "adam", device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        input_size : int
            Size of input
        output_size : int
            Size of output
        learning_rate : float
            Learning rate of model
        op_algorithm : str, optional
            Optimization algorithm, by default adam
        device_name : str, optional
            Device to run tensors in
        """
        super(NineLayers, self).__init__("Nine Layers NN", learning_rate,
                                         op_algorithm=op_algorithm, device_name=device_name)
        self.__model__ = FullNineLayers(input_size, output_size, self.__device__).double()
        self.__optimizer__ = self.__optimizer_callable__(self.__model__.parameters(), lr=self.__lr__)
        return
