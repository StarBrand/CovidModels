""" Plot Contribution class """

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from accessories.config import CONFIG


class ContributionPlotter:
    """
    Contribution Plotter abstract class
    """
    def __init__(self) -> None:
        return

    @staticmethod
    def plot_saliency(saliency: pd.DataFrame, path_to_save: str = None, axes: Axes = None) -> None:
        """
        Plots saliency and saved on disk

        Parameters
        ----------
        saliency : DataFrame
            Data to plot saliency map, it must contain:
            Saliency (float), chr (int) and pos (int)
        path_to_save : str
            Path to save plot as image and summary of top 10
        axes : Axes
            Axes to plot saliency

        Returns
        -------
        None
        """
        ContributionPlotter.__plot_contrition__(saliency, "Saliency", path_to_save, axes=axes)
        return

    @staticmethod
    def plot_wald(wald: pd.DataFrame, path_to_save: str = None, axes: Axes = None) -> None:
        """
        Plots Walt test and saved on disk

        Parameters
        ----------
        wald : DataFrame
            Data to plot gwas map, it must contain:
            Wald (float), chr (int) and pos (int)
        path_to_save : str
            Path to save plot as image and summary of top 10
        axes : Axes
            Axes to plot wald statistics

        Returns
        -------
        None
        """
        wald.rename(columns={"wald": "Wald"}, inplace=True)
        ContributionPlotter.__plot_contrition__(wald, "Wald", path_to_save, axes=axes)
        return

    @staticmethod
    def plot_p_value(p_value: pd.DataFrame,
                     path_to_save: str = None,
                     plot_significance: bool = False,
                     axes: Axes = None) -> None:
        """
        Plots P value from glm analysis and saved n disk

        Parameters
        ----------
        p_value : DataFrame
            Data to plot gwas, it must contain:
            Pr(>|z|) (float), chr (int) and pos (int)
        path_to_save : str
            Path to save plot as image and summary of top 10
        plot_significance : bool
            Plot significance line (suggestive: 6e-5, significance: 5e-8)
        axes : Axes
            Axes to plot -log(p)

        Returns
        -------
        None
        """
        p_value["-log(p)"] = -np.log10(p_value["Pr(>|z|)"])
        ContributionPlotter.__plot_contrition__(p_value, "-log(p)", path_to_save, plot_significance, axes=axes)
        return

    @staticmethod
    def __plot_contrition__(
            data: pd.DataFrame, contribution: str, path_to_save: str = None,
            plot_significance: bool = False, axes: Axes = None) -> None:
        """
        Plots contribution and saved on disk

        Parameters
        ----------
        data : DataFrame
            Data to plot contribution map, it must contain:
            <contribution> (float), chr (int) and pos (int)
        contribution : str
            Name of contribution
        path_to_save : str
            Path to save plot as image and summary of top 10
        plot_significance : bool
            Plot significance line (suggestive: 6e-5, significance: 5e-8),
            just if significance is visualize with p-value
        axes : Axes
            Axes to plot contribution

        Returns
        -------
        None
        """
        whole_data = data.copy(deep=True)
        whole_data.sort_values(["chr", "pos"], inplace=True)
        one_chr = len(pd.unique(whole_data["chr"])) == 1
        if "ind" not in whole_data.columns:
            whole_data = ContributionPlotter.generate_independent_position(whole_data)
        grouped_data = whole_data.groupby("chr")
        if axes is None:
            fig_size = CONFIG["FIG_SIZE"]
            fig_size = (fig_size[0] * 2, fig_size[1] * 2)
            fig, axes = plt.subplots(figsize=fig_size)
        else:
            fig = None
            if path_to_save is not None:
                logging.warning("If Axis given, figure will not be saved, ignorig path_to_save argument")
        tick_labels = list()
        tick_position = list()
        for i, (chromosome, chr_data) in enumerate(grouped_data):
            logging.info("Plotting chromosome {} with {} variants".format(chromosome, len(chr_data)))
            chr_data.plot(
                kind="scatter", x="ind",
                y=contribution, color=CONFIG["CHROMOSOME_COLORS"][i % 2], ax=axes
            )
            first_one = chr_data["ind"].iloc[0]
            last_one = chr_data["ind"].iloc[-1]
            if one_chr:
                logging.info("Plotting one chromosome")
                tick_labels = np.linspace(first_one, last_one, 5, endpoint=True).astype(int)
                tick_position = tick_labels
            else:
                tick_labels.append(chromosome)
                tick_position.append((last_one + first_one) / 2)

        if contribution == "-log(p)" and plot_significance:
            axes.axhline(-np.log10(1e-5), color="b")
            axes.axhline(-np.log10(5e-8), color="r")

        axes.margins(0.01)
        axes.set_xticks(tick_position)
        axes.set_xticklabels(tick_labels)
        axes.tick_params(which="both", top=False)
        axes.xaxis.set_label_position('bottom')
        axes.xaxis.set_ticks(tick_position)
        axes.set_xticklabels(tick_labels, fontsize=CONFIG["TICK_SIZE"])
        axes.tick_params(axis="y", labelsize=CONFIG["TICK_SIZE"])
        axes.set_xlabel("Chromosomes" if not one_chr else "Position", fontsize=CONFIG["LABEL_SIZE"] * 1.5)
        axes.set_ylabel(contribution, fontsize=CONFIG["LABEL_SIZE"] * 1.5)
        if fig is not None:
            fig.tight_layout()
        if path_to_save is not None and fig is not None:
            plot_path = path_to_save + ".png"
            logging.info("Saving plot on {}".format(plot_path))
            fig.savefig(plot_path)
        return None

    @staticmethod
    def qq_plot(data: pd.DataFrame, path_to_save: str) -> None:
        """
        Generates a Quantile-Quantile Plot and save it on `path_to_save`

        Parameters
        ----------
        data : DataFrame
            Data with p-value for qq plot
        path_to_save : str
            Path to save graphs

        Returns
        -------
        None
        """
        whole_data = data.copy(deep=True)
        observed = -np.log10(sorted(whole_data["Pr(>|z|)"]))
        n = len(observed)
        a = 1 / 2 if n > 10 else 3 / 8
        expected = np.linspace(a, n - a, n) / (n + 1 - (2 * a))
        expected = -np.log10(expected)
        fig_size = CONFIG["MATRIX_SIZE"]
        fig, ax = plt.subplots(figsize=fig_size)
        ax.axline((0, 0), slope=1, color="r")
        ax.plot(expected, observed, "o", color="k")
        ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=CONFIG["LABEL_SIZE"])
        ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=CONFIG["LABEL_SIZE"])
        ax.tick_params(which="both", top=False)
        ax.tick_params(axis="x", labelsize=CONFIG["TICK_SIZE"])
        ax.tick_params(axis="y", labelsize=CONFIG["TICK_SIZE"])
        plot_path = path_to_save + ".png"
        logging.info("Saving plot on {}".format(plot_path))
        fig.tight_layout()
        fig.savefig(plot_path)
        return

    @staticmethod
    def generate_independent_position(data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates independent position to use on Manhattan plot

        Parameters
        ----------
        data : DataFrame
            Data to plot contribution map, it must contain:
            <contribution> (float), chr (int) and pos (int)

        Returns
        -------
        DataFrame
            Data with ind (int) column to serve as position on Manhattan Plot
        """
        out = data.copy(deep=True)
        out.sort_values(["chr", "pos"], inplace=True)
        out["ind"] = out["pos"]
        current_pos = 0
        for chromosome in np.sort(pd.unique(out["chr"])):
            out.loc[out[out["chr"] == chromosome].index, "ind"] += current_pos
            current_pos += out[out["chr"] == chromosome]["pos"].max()
        return out
