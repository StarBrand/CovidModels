"""amphi_models.py: Models that use both, clinical and genetical data """

from __future__ import annotations
import torch
import pandas as pd
from abc import ABC, abstractmethod
from models import ConvFullyNeuralNetwork
from models.architectures import ModifiedModule, Soybean
from models.architectures import PartialTwoLayers, PartialThreeLayers, PartialFiveLayers, PartialNineLayers


LAYERS = {
    2: (PartialTwoLayers, 500),
    3: (PartialThreeLayers, 200),
    5: (PartialFiveLayers, 500),
    9: (PartialNineLayers, 500),
}


class AmphiModel(ConvFullyNeuralNetwork, ABC):
    """
    Architectures that use clinical data as well
    """
    class TheModel(ModifiedModule):
        """
        The model that use genetic and clinical data
        """
        def __init__(
                self,
                genome_model: torch.nn.Module,
                last_features: int,
                covariates: int,
                output_size: int,
                device: torch.device,
                layers: int = 3
        ) -> None:
            super(AmphiModel.TheModel, self).__init__(device)
            self.__covariates__ = covariates
            self.__device__: torch.device = device
            self.__genome_model__: torch.nn.Module = genome_model
            try:
                self.__clinical_model__ = LAYERS[layers][0](covariates, device)
            except KeyError:
                if type(layers) == int:
                    raise NotImplemented("{} layers of clinical model not implemented".format(layers))
                else:
                    raise TypeError("{} not a valid argument of layers for clinical model".format(layers))
            self.__perceptron__ = torch.nn.Sequential(
                torch.nn.Linear(last_features + LAYERS[layers][1], output_size),
            ).float()
            self.__register_module__("clinical_model")
            self.__register_module__("genome_model")
            self.__register_module__("perceptron")
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
            features_genetic = self.__genome_model__(x[..., 0:-self.__covariates__])
            features_clinical = self.__clinical_model__(x[..., 0, -self.__covariates__:])
            features = torch.cat((features_genetic, features_clinical), dim=1)
            return self.__perceptron__(features)

    @abstractmethod
    def __init__(
            self, name: str, genome_size: int, channels: int,
            covariates: int, output_size: int, learning_rate: float,
            weight: torch.Tensor = None, device_name: str = None
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        name : str
            Name for reports
        genome_size : int
            Size of genome data
        channels : int
            Number of channels
        covariates : int
            Number of covariates
        output_size : int
            Number of classes to predict
        learning_rate : float
            Learning rate of NN
        weight : torch.Tensor
            Weight for each class for loss function
        device_name : str, optional
            Device to run tensors in
        """
        super(AmphiModel, self).__init__(
            name, genome_size, channels, output_size,
            learning_rate, weight=weight, device_name=device_name,
        )
        self.__covariates__ = covariates
        return

    def as_tensor(self, x_train: pd.DataFrame) -> torch.tensor:
        """
        Convert train set from dataframe to tensor
        separating clinical and genetic variables

        Parameters
        ----------
        x_train : DataFrame
            Train set

        Returns
        -------
        tensor
            Train set as tensor
        """
        as_numeric = torch.tensor(x_train.to_numpy())
        x_genome = as_numeric[..., 0:-self.__covariates__]
        x_clinical = as_numeric[..., -self.__covariates__:]
        complete_zeros = torch.zeros(x_clinical.shape[0], 1, x_clinical.shape[1])
        x_clinical = torch.cat(
            [x_clinical.reshape(x_clinical.shape[0], 1, x_clinical.shape[1])] +
            [complete_zeros] * (self.__channels__ - 1), dim=1
        )
        return torch.cat((
            super(AmphiModel, self).one_hot(x_genome, self.__channels__).float(),
            x_clinical.float()
        ), dim=2).float()


class AmphiSoybean(AmphiModel):
    """
    Soybean architecture that receives clinical data as covariates
    """
    def __init__(self, genome_size: int, channels: int, covariates: int, output_size: int,
                 learning_rate: float, weight_decay: float = 0.0,
                 layers: int = 3, weight: torch.Tensor = None,
                 device_name: str = None, dropout_p: float = 0.75) -> None:
        super(AmphiSoybean, self).__init__(
            "Extended Dual-stream CNN", genome_size,
            channels, covariates, output_size,
            learning_rate, weight=weight,
            device_name=device_name
        )
        self.__soybean_model__ = Soybean(
            genome_size, channels, self.__device__, dropout_p=dropout_p
        )
        self.__model__ = AmphiModel.TheModel(
            self.__soybean_model__,
            self.__soybean_model__.__input_fnn__,
            covariates,
            output_size,
            layers=layers,
            device=self.__device__
        )
        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=self.__lr__, weight_decay=weight_decay)
