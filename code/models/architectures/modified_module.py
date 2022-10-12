""" Modified nn.Module to ease architecture implementation """

from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import List


class ModifiedModule(torch.nn.Module, ABC):
    """
    Modified module to facilitate architectures implementation
    """
    @abstractmethod
    def __init__(self, device: torch.device) -> None:
        """
        Constructor

        Parameters
        ----------
        device : torch.device
            Device to locate the model
        """
        super(ModifiedModule, self).__init__()
        self.__device__: torch.device = device
        self.__modules__: List[str] = list()
        return

    def __register_module__(self, module: str) -> None:
        """
        Registers module with a name to manipulate during
        allocation on devices

        Parameters
        ----------
        module : str
            Module name

        Returns
        -------
        None, alter self.__modules__
        """
        name = "__{}__".format(module)
        if name not in self.__modules__:
            self.__modules__.append(name)

    @abstractmethod
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
        pass

    def to(self, new_device: torch.device) -> ModifiedModule:
        """
        Overwritten of `to`, to move model to a device

        Parameters
        ----------
        new_device : device
            Device to move model

        Returns
        -------
        TheModel
            Pass model to a new device and return a reference to self
        """
        for module in self.__modules__:
            getattr(self, module).to(new_device)
        return self

    def cpu(self) -> ModifiedModule:
        """
        Overwritten of `cpu`, to move model to cpu

        Returns
        -------
        TheModel
            Pass model to a cpu and return a reference to self
        """
        self.__device__ = torch.device("cpu")
        self.to(self.__device__)
        return self

    def double(self) -> ModifiedModule:
        """
        Double wrapper

        Returns
        -------
        ModifiedModule
            double version
        """
        for module in self.__modules__:
            getattr(self, module).double()
        return self

    def float(self) -> ModifiedModule:
        """
        Float wrapper

        Returns
        -------
        ModifiedModule
            float version
        """
        for module in self.__modules__:
            getattr(self, module).float()
        return self
