""" Usefull methods """

import math
from typing import Tuple, List
from Bio import Entrez
from xml.etree import ElementTree


def show_percentage_of_process(total_size: int, end_size: int) -> str:
    """
    Shows a bar with visualization advance

    Parameters
    ----------
    total_size : int
        Total size of process
    end_size : int
        Current iteration of process

    Returns
    -------
    str
        A text indicating current iteration of process
    """
    one_fifth = total_size / 50
    to_show = math.ceil(end_size / one_fifth)
    digits = math.floor(math.log10(total_size)) + 1
    return ("|"*to_show + "-"*(50 - to_show) + " => {:0%dd}/{:0%dd}   " % (digits, digits)).format(
        end_size, total_size
    )


def get_gene_rsid(chromosome: int, position: int) -> Tuple[List[str], str]:
    """
    Executes query to https://eutils.ncbi.nlm.nih.gov/entrez and retrieve genes and rsid


    Parameters
    ----------
    chromosome : int
        Chromosome of SNP
    position : int
        Position of SNP

    Returns
    -------
    List[str]
        Genes on SNPs
    str
        RSid of SNP
    """
    first_handle = Entrez.esearch(
        db="snp",
        term="{:d}[chr] AND {:d}[position]".format(chromosome, position),
        usehistory="y",
        retmax=1
    )
    first_result = Entrez.read(first_handle)
    second_result = Entrez.efetch(
        db="snp",
        retmode="xml",
        retmax=2,
        webenv=first_result["WebEnv"],
        query_key=first_result["QueryKey"]
    )
    data = second_result.read()
    second_result.close()
    data = data.decode()
    tree = ElementTree.fromstring(data)
    genes = list()
    rsid = ""
    for child in tree[0]:
        if "GENES" in child.tag:
            for grandchild in child:
                for gene in grandchild:
                    if "NAME" in gene.tag:
                        genes.append(gene.text)
        elif "SNP_ID" in child.tag and "SORT" not in child.tag:
            rsid = "rs{}".format(child.text)
    return genes, rsid
