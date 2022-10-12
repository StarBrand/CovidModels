""" One Hot Encoder Wrapper and normalized numerical columns """

import pandas as pd
from typing import List


class Encoder:
    """
    Encoder class. to execute One Hot Encoding and normalization
    """
    class Normalizer:
        """
        Normalizer class, normalize numerical columns
        """
        def __init__(self, data: pd.DataFrame) -> None:
            """
            Class constructor

            Parameters
            ----------
            data : DataFrame
                Data to normalize
            """
            self.__min__ = data.min(axis=0)
            self.__max__ = data.max(axis=0)
            return

        def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
            """

            Parameters
            ----------
            data

            Returns
            -------

            """
            out_numerical = pd.DataFrame([], index=data.index)
            for column in data.columns:
                out_numerical[column] = pd.Series((data[column] - self.__min__[column]
                                                   ) / (self.__max__[column] - self.__min__[column]),
                                                  index=out_numerical.index)
            return out_numerical

        def decode(self, data: pd.DataFrame) -> pd.DataFrame:
            """

            Parameters
            ----------
            data

            Returns
            -------

            """
            out_numerical = pd.DataFrame([], index=data.index)
            for column in data.columns:
                out_numerical[column] = pd.Series(
                    data[column] * (self.__max__[column] - self.__min__[column]) + self.__min__[column],
                    index=out_numerical.index)
            return out_numerical

    def __init__(self, data: pd.DataFrame, numerical: List[str]) -> None:
        """
        Class constructor

        Parameters
        ----------
        data : DataFrame
            Input data
        numerical : List[str]
            Names of numerical columns
        """
        self.__numerical__ = list(set(numerical.copy()).intersection(data.columns))
        self.__normalizer__ = Encoder.Normalizer(data[self.__numerical__])
        self.__not_numerical__ = list(filter(lambda x: x not in self.__numerical__, data.columns))
        return

    def encode(self, data: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Encode data provided. Note data provided have to be a subset of the data given
        as argument in the constructor

        Parameters
        ----------
        data : DataFrame
            Input data
        normalize : bool, optional
            Normalize numerical values

        Returns
        -------
        DataFrame:
            Data encoded
        """
        out_x = pd.get_dummies(data.drop(columns=self.__numerical__)) * 1.0
        if normalize:
            out_numerical = self.__normalizer__.normalize(data[self.__numerical__])
        else:
            out_numerical = data[self.__numerical__].copy(deep=True)
        return pd.concat([out_x, out_numerical], axis=1)

    def decode(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Decodes data to original dataframe

        Parameters
        ----------
        data : DataFrame
            DataFrame to decode

        Returns
        -------
        DataFrame
            Data in the original format
        """
        dump_columns = list(filter(lambda x: x not in self.__numerical__, data.columns))
        original_index = list(data.index)
        decoded_data = pd.DataFrame({}, columns=self.__not_numerical__, index=original_index)
        for column in self.__not_numerical__:
            particular_dump_columns = list(filter(lambda x: column in x, dump_columns))
            for encoded_col in particular_dump_columns:
                value = encoded_col.replace("{}_".format(column), "")
                decoded_data.loc[
                    data[data[encoded_col] == 1].index,
                    column
                ] = value
        decoded_numerical = self.__normalizer__.decode(
            data[set(self.__numerical__).intersection(data.columns)])
        return pd.concat([decoded_data, decoded_numerical], axis=1)
