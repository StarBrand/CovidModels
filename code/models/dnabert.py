""" DNABERT Class wrapper """

import gc
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from transformers import AutoTokenizer
from transformers import BertModel
from models import NeuralNetwork
from models.architectures import ModifiedModule
from models.amphi_models import LAYERS
from accessories import show_percentage_of_process


class DNABert(NeuralNetwork):
    """
    DNABert Wrapper to work as models in project.
    This is implemented by using
    [HuggingFace](https://huggingface.co/docs/transformers/index)

    DNABert GitHub Repository:
    https://github.com/jerryji1993/DNABERT

    DNABert publication:
    https://doi.org/10.1093/bioinformatics/btab083
    """
    URL = "armheb/DNA_bert_6"

    class DNABertModel(ModifiedModule):
        """
        DNABert Model uses for classification
        """
        def __init__(self, output_size: int, device: torch.device) -> None:
            super(DNABert.DNABertModel, self).__init__(device)
            self.__attention__: List[torch.tensor] = list()
            self.__bert__ = BertModel.from_pretrained(DNABert.URL, output_attentions=True)
            self.__drop__ = torch.nn.Dropout(p=0.3)
            self.__dense__ = torch.nn.Linear(self.__bert__.config.hidden_size, output_size)
            self.__register_module__("bert")
            self.__register_module__("drop")
            self.__register_module__("dense")
            return

        def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
            """
            Forward function for DNABERT

            Parameters
            ----------
            x : tensor
                Input ids

            Other Parameters
            ----------------
            token_type_ids : tensor
                Token types
            attention_mask : tensor
                Attention mask

            Returns
            -------
            tensor
                Output
            """
            bert_output = self.__bert__(
                x, token_type_ids=kwargs.get("token_type_ids"),
                attention_mask=kwargs.get("attention_mask")
            )
            self.__attention__ = [attention.clone().detach().cpu() for attention in bert_output.attentions]
            return self.__dense__(self.__drop__(bert_output.pooler_output))

    def __init__(self,
                 output_size: int,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 epsilon: float = 1e-08,
                 weight: torch.Tensor = None,
                 device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        learning_rate : float
            Learning rate for training
        """
        super(DNABert, self).__init__(
            "DNA BERT", learning_rate=learning_rate,
            weight=weight,
            device_name=device_name
        )
        self.__tokenizer__ = AutoTokenizer.from_pretrained(
            DNABert.URL
        )
        self.__model__ = DNABert.DNABertModel(output_size, self.__device__)
        self.__optimizer__ = torch.optim.AdamW(
            self.__model__.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=epsilon
        )
        return

    def __call__(self, *args, **kwargs) -> torch.tensor:
        """
        Calling method

        Parameters
        ----------
        args[0] : DataFrame
            input data

        Other Parameters
        ----------------
        cpu : bool
            Calculate on cpu and detached
        
        Returns
        -------
        tensor
            Output
        """
        def tokenizer_function(example: str) -> Tuple[List[int], List[int], List[int]]:
            """
            Tokenizer function
            Parameters
            ----------
            example : str
                Example test sequence as 6-mer
            Returns
            -------
            List[int]
                Tokens id
            List[int]
                Token type
            List[int]
                Attention mask
            """
            output = self.__tokenizer__(example)
            return output["input_ids"], output["token_type_ids"], output["attention_mask"]

        x_train = args[0].copy(deep=True)
        x_train["input_ids"], x_train["token_type_ids"], x_train["attention_mask"] = zip(
            *x_train["sequence"].map(tokenizer_function)
        )
        if kwargs.get("cpu", False):
            return self.__model__(
                torch.tensor(x_train["input_ids"]).long().cpu(),
                token_type_ids=torch.tensor(x_train["token_type_ids"]).long().cpu(),
                attention_mask=torch.tensor(x_train["attention_mask"]).long().cpu()
            )
        else:
            return self.__model__(
                torch.tensor(x_train["input_ids"]).long().to(self.__device__),
                token_type_ids=torch.tensor(x_train["token_type_ids"]).long().to(self.__device__),
                attention_mask=torch.tensor(x_train["attention_mask"]).long().to(self.__device__)
            )

    def get_attention(self, x_test: pd.DataFrame, kmer: int = 6, batch_size: int = 20) -> pd.DataFrame:
        """
        Gets attention for the test set

        Parameters
        ----------
        x_test : DataFrame
            Test set attributes
        kmer : int, optional
            k, for tokens in k-mer format, default: 6
        batch_size : int, optional
            Size of batch to calculate attention, default: 20

        Returns
        -------
        DataFrame
            Attention matrix as participant per nucleotide
        """
        def get_attention_score(attentions: List[torch.tensor]) -> np.ndarray:
            """
            Get attention score from attentions in output

            Parameters
            ----------
            attentions : tensor
                Attentions in output

            Returns
            -------
            ND-Array
                Get attention score for each participants and nucleotide
            """
            def get_score_per_nucleotide(att_scores: np.ndarray) -> np.ndarray:
                """
                Gets score per nucleotide from attention score per token

                Parameters
                ----------
                att_scores : ND-Array
                    Attention scores per token

                Returns
                -------
                ND-Array
                    Attention per nucleotide
                """
                tokens, participants = att_scores.shape
                nucleotides = tokens + kmer - 1
                counts = np.zeros((nucleotides, participants))
                real_scores = np.zeros((nucleotides, participants))
                for ix, score in enumerate(att_scores):
                    for k in range(kmer):
                        counts[ix+k, :] += 1.0
                        real_scores[ix+k, :] += score
                return real_scores / counts

            attention = torch.stack([layer_attention for layer_attention in attentions])
            attn_score = attention[11, :, :, 0, :].sum(dim=1).transpose(0, 1).detach().numpy()
            first_haplotype = attn_score[1:255, :]
            second_haplotype = attn_score[256:-1, :]
            return np.concatenate((
                get_score_per_nucleotide(first_haplotype),
                get_score_per_nucleotide(second_haplotype)
            ), axis=0)

        output = list()
        with torch.no_grad():
            self.__model__.eval()
            i = 0
            j = batch_size
            go_on = True
            while go_on:
                if j > len(x_test):
                    j = len(x_test)
                if j == len(x_test):
                    go_on = False
                print(show_percentage_of_process(len(x_test), j), end="\r")
                current_test = x_test.iloc[i:j, :]
                _ = self(current_test, cpu=True)
                attn_array = get_attention_score(self.__model__.__attention__)
                output.append(pd.DataFrame(attn_array.transpose(), index=current_test.index))
                del attn_array, self.__model__.__attention__, current_test
                gc.collect()
                i += batch_size
                j += batch_size

        return pd.concat(output)


class AmphiDNABert(DNABert):
    """
    DNABert extended to use clinical data as covariates
    """
    class DNABertAmphiModel(ModifiedModule):
        """
        DNABert Model uses for classification
        """
        def __init__(self, covariates: int, output_size: int, device: torch.device, layers: int = 3) -> None:
            super(AmphiDNABert.DNABertAmphiModel, self).__init__(device)
            self.__attention__: List[torch.tensor] = list()
            self.__bert__ = BertModel.from_pretrained(DNABert.URL, output_attentions=True)
            self.__drop__ = torch.nn.Dropout(p=0.3)
            try:
                self.__clinical_model__ = LAYERS[layers][0](covariates, device)
            except KeyError:
                if type(layers) == int:
                    raise NotImplemented("{} layers of clinical model not implemented".format(layers))
                else:
                    raise TypeError("{} not a valid argument of layers for clinical model".format(layers))
            self.__perceptron__ = torch.nn.Sequential(
                torch.nn.Linear(self.__bert__.config.hidden_size + LAYERS[layers][1], output_size),
            ).float()
            self.__register_module__("bert")
            self.__register_module__("drop")
            self.__register_module__("clinical_model")
            self.__register_module__("perceptron")
            return

        def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
            """
            Forward function for DNABERT

            Parameters
            ----------
            x : tensor
                Input ids

            Other Parameters
            ----------------
            token_type_ids : tensor
                Token types
            attention_mask : tensor
                Attention mask
            covariates : tensor
                Clinical data as covariates

            Returns
            -------
            tensor
                Output
            """
            bert_output = self.__bert__(
                x, token_type_ids=kwargs.get("token_type_ids"),
                attention_mask=kwargs.get("attention_mask")
            )
            self.__attention__ = [attention.clone().detach().cpu() for attention in bert_output.attentions]
            bert_features = self.__drop__(bert_output.pooler_output)
            clinical_features = self.__clinical_model__(kwargs.get("covariates"))
            return self.__perceptron__(torch.cat((bert_features, clinical_features), dim=1))

    def __init__(self,
                 covariates: int,
                 output_size: int,
                 layers: int = 3,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 epsilon: float = 1e-08,
                 weight: torch.Tensor = None,
                 device_name: str = None) -> None:
        """
        Constructor

        Parameters
        ----------
        covariates : int
            Features of clinical data
        layers : int
            Number of layers of clinical model
        weight : torch.Tensor, optional
            Weight for each class for loss function
        learning_rate : float
            Learning rate for training
        """
        super(DNABert, self).__init__(
            "Extended DNA BERT",
            learning_rate=learning_rate,
            weight=weight,
            device_name=device_name
        )
        self.__tokenizer__ = AutoTokenizer.from_pretrained(
            DNABert.URL
        )
        self.__model__ = AmphiDNABert.DNABertAmphiModel(
            covariates, output_size,
            self.__device__, layers=layers
        )
        self.__optimizer__ = torch.optim.AdamW(
            self.__model__.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=epsilon
        )
        return

    def __call__(self, *args, **kwargs) -> torch.tensor:
        """
        Calling method

        Parameters
        ----------
        args[0] : DataFrame
            input data

        Other Parameters
        ----------------
        cpu : bool
            Calculate on cpu and detached

        Returns
        -------
        tensor
            Output
        """
        def tokenizer_function(example: str) -> Tuple[List[int], List[int], List[int]]:
            """
            Tokenizer function
            Parameters
            ----------
            example : str
                Example test sequence as 6-mer
            Returns
            -------
            List[int]
                Tokens id
            List[int]
                Token type
            List[int]
                Attention mask
            """
            output = self.__tokenizer__(example)
            return output["input_ids"], output["token_type_ids"], output["attention_mask"]

        x_train = args[0].copy(deep=True)
        x_train["input_ids"], x_train["token_type_ids"], x_train["attention_mask"] = zip(
            *x_train["sequence"].map(tokenizer_function)
        )
        if kwargs.get("cpu", False):
            return self.__model__(
                torch.tensor(x_train["input_ids"]).long().cpu(),
                token_type_ids=torch.tensor(x_train["token_type_ids"]).long().cpu(),
                attention_mask=torch.tensor(x_train["attention_mask"]).long().cpu(),
                covariates=torch.tensor(args[0].drop(columns=["sequence"]).to_numpy()).float().cpu()
            )
        else:
            return self.__model__(
                torch.tensor(x_train["input_ids"]).long().to(self.__device__),
                token_type_ids=torch.tensor(x_train["token_type_ids"]).long().to(self.__device__),
                attention_mask=torch.tensor(x_train["attention_mask"]).long().to(self.__device__),
                covariates=torch.tensor(args[0].drop(columns=["sequence"]).to_numpy()).float().to(self.__device__)
            )
