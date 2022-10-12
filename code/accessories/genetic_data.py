""" Import genetic data """

import logging
import pandas as pd
from typing import List
from pyplink import PyPlink
from pandas_plink import read_plink1_bin


class GeneticData:
    """
    Genetic Data class, store and give access to genetic data

    Attributes
    ----------
    __x_data_array__ : xarray.DataArray
        xDataArray with genetic data for easy access
    __plink_file__ : PyPlink
        PyPlink file with whole genetic data
    __filtered__ : bool
        Whether or not this was filtered (by calling .apply_filter())
    """
    def __init__(self, path: str) -> None:
        """
        Constructor, with the path of plink format files

        Parameters
        ----------
        path : str
            Path to plink files
        """
        self.__x_data_array__ = read_plink1_bin(
            "{}.bed".format(path),
            "{}.bim".format(path),
            "{}.fam".format(path),
        )
        self.__plink_file__ = PyPlink(path)
        self.__filtered__ = False
        return

    def apply_filter(self, chromosomes: List[str] = None,
                     snp: List[str] = None,
                     positions: List[int] = None,
                     max_snp: int = None) -> None:
        """
        Applies filter to snp

        Parameters
        ----------
        chromosomes : List[str], optional
            Filter by some chromosomes
        snp : List[str], optional
            Filter by snp names
        positions : List[int], optional
            Filter by position of snp
        max_snp : int, optional
            Get just max_snp variants

        Returns
        -------
        None, alter __x_data_array__
        """
        if chromosomes is not None:
            self.__x_data_array__ = self.__x_data_array__.where(
                self.__x_data_array__.chrom.isin(chromosomes),
                drop=True
            )
            self.__filtered__ = True
        if snp is not None:
            self.__x_data_array__ = self.__x_data_array__.where(
                self.__x_data_array__.snp.isin(snp),
                drop=True
            )
            self.__filtered__ = True
        if positions is not None:
            self.__x_data_array__ = self.__x_data_array__.where(
                self.__x_data_array__.pos.isin(positions),
                drop=True
            )
            self.__filtered__ = True
        if max_snp is not None:
            all_snp = pd.unique(self.__x_data_array__.snp.to_pandas())
            selected_snp = all_snp[0:max_snp]
            self.__x_data_array__ = self.__x_data_array__.where(
                self.__x_data_array__.snp.isin(selected_snp),
                drop=True
            )
            self.__filtered__ = True
        return

    def get_values(self) -> pd.DataFrame:
        """
        Get SNP as values ({0, 1, 2} matrix)

        Returns
        -------
        DataFrame
            A pandas DataFrame with matrix of {0, 1, 2}
        """
        return pd.DataFrame(
            self.__x_data_array__.values,
            index=self.__x_data_array__.sample.to_pandas(),
            columns=self.__x_data_array__.variant.snp.to_pandas(),
        )

    def get_genotype(self, max_snp: int = None) -> pd.DataFrame:
        """
        Get genotypes as two alleles string

        Parameters
        ----------
        max_snp : int, optional
            Maximum number of variant to get

        Returns
        -------
        DataFrame
            A pandas DataFrame wih matrix of b1b2
            With b1 in {A, C, T, G, 0}
            and b2 in {A, C, T, G, 0}
        """
        # Rebuild matrix as pandas:
        all_snp = list()
        snp_genotype = list()

        i = 0
        if self.__filtered__:
            genotype_generator = self.__plink_file__.iter_acgt_geno_marker(
                pd.unique(self.__x_data_array__.variant.snp.to_pandas())
            )
        else:
            genotype_generator = self.__plink_file__.iter_acgt_geno()
        for marker, genotype in genotype_generator:
            i += 1
            if i % 100 == 0:
                logging.debug("Markers imported: {}".format(i))
            all_snp.append(marker)
            snp_genotype.append(genotype)
            if max_snp is not None and i == max_snp:
                break
        samples = self.__plink_file__.get_fam()["iid"]
        data = pd.DataFrame(snp_genotype, index=all_snp).transpose()
        data.index = samples
        return data
