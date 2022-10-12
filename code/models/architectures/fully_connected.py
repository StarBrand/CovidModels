""" Fully connected architectures """

import torch
from abc import ABC, abstractmethod
from models.architectures import ModifiedModule


class _PartialFully(ModifiedModule, ABC):
    """
    Two Layers without final step
    """
    @abstractmethod
    def __init__(self, device: torch.device) -> None:
        super(_PartialFully, self).__init__(device)
        self.__main__: ModifiedModule
        self.__register_module__("main")
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
        return self.__main__(x)


class _FullFully(ModifiedModule, ABC):
    """
    Two Layers without final step
    """
    @abstractmethod
    def __init__(self, device: torch.device) -> None:
        super(_FullFully, self).__init__(device)
        self.__main__: ModifiedModule
        self.__end_layer__: ModifiedModule
        self.__register_module__("main")
        self.__register_module__("end_layer")
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
        out = self.__main__(x)
        return self.__end_layer__(out)


class PartialTwoLayers(_PartialFully):
    """
    Two Layers without final step
    """
    def __init__(self, input_size: int, device: torch.device) -> None:
        super(PartialTwoLayers, self).__init__(device)
        self.__main__ = torch.nn.Sequential(
            torch.nn.Linear(input_size, 500),
            torch.nn.ReLU(),
        )
        return


class FullTwoLayers(_FullFully):
    """
    Two Layers without final step
    """
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(FullTwoLayers, self).__init__(device)
        self.__main__ = PartialTwoLayers(input_size, device)
        self.__end_layer__ = torch.nn.Sequential(
            torch.nn.Linear(500, output_size),
        )
        return


class PartialThreeLayers(_PartialFully):
    """
    Three Layers without final step
    """
    def __init__(self, input_size: int, device: torch.device) -> None:
        super(PartialThreeLayers, self).__init__(device)
        self.__main__ = torch.nn.Sequential(
            torch.nn.Linear(input_size, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.Tanh(),
        )
        return


class FullThreeLayers(_FullFully):
    """
    Three Layers without final step
    """
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(FullThreeLayers, self).__init__(device)
        self.__main__ = PartialThreeLayers(input_size, device)
        self.__end_layer__ = torch.nn.Sequential(
            torch.nn.Linear(200, output_size),
        )
        return


class PartialFiveLayers(_PartialFully):
    """
    Five Layers without final step
    """
    def __init__(self, input_size: int, device: torch.device) -> None:
        super(PartialFiveLayers, self).__init__(device)
        self.__main__ = torch.nn.Sequential(
            torch.nn.Linear(input_size, 500),
            torch.nn.ReLU(),
            *[torch.nn.Linear(500, 500),
              torch.nn.ReLU()] * 3,
        )
        return


class FullFiveLayers(_FullFully):
    """
    Five Layers without final step
    """
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(FullFiveLayers, self).__init__(device)
        self.__main__ = PartialFiveLayers(input_size, device)
        self.__end_layer__ = torch.nn.Sequential(
            torch.nn.Linear(500, output_size),
        )
        return


class PartialNineLayers(_PartialFully):
    """
    Nine Layers without final step
    """
    def __init__(self, input_size: int, device: torch.device) -> None:
        super(PartialNineLayers, self).__init__(device)
        self.__main__ = torch.nn.Sequential(
            torch.nn.Linear(input_size, 500),
            torch.nn.ReLU(),
            *[torch.nn.Linear(500, 500),
              torch.nn.ReLU()] * 7,
        )
        return


class FullNineLayers(_FullFully):
    """
    Nine Layers without final step
    """
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(FullNineLayers, self).__init__(device)
        self.__main__ = PartialNineLayers(input_size, device)
        self.__end_layer__ = torch.nn.Sequential(
            torch.nn.Linear(500, output_size),
        )
        return
