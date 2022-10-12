""" Architectures implemented in PyTorch """

from models.architectures.modified_module import ModifiedModule
from models.architectures.fully_connected import PartialTwoLayers, FullTwoLayers
from models.architectures.fully_connected import PartialThreeLayers, FullThreeLayers
from models.architectures.fully_connected import PartialFiveLayers, FullFiveLayers
from models.architectures.fully_connected import PartialNineLayers, FullNineLayers
from models.architectures.convultional import ConvolutionalReduction, ConvolutionalFully
from models.architectures.convultional import Soybean, SoybeanFully
from models.architectures.recurrent import RecurrentNN
from models.architectures.recurrent import LongShortTermMemory
