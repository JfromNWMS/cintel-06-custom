import pandas as pd
import matplotlib.pyplot as plt
from pingouin import qqplot
from scipy.stats import shapiro
from math import ceil, sqrt


def is_norm_small(df_data: pd.DataFrame, by: str = None, ncol_fig: int = None, main_label: bool = True, confidence: float = 0.95) -> None:
    """
    Function is_norm_small() takes in a pandas.DataFrame and plots normal Q-Q plots 
    for all columns of the dataframe in square configuration. The Shapiro-Wilk normality
    test statistic and p-value are also calculated for each plot and placed in the upper 
    left corner of the plot.

    Args:
        df_data (pd.DataFrame): Input dataframe whose columns will be plotted over.
        by (str): Name of categorical column in dataframe to plot by. Defaults to no categorical column.
        ncol_fig (int): Number of columns to force the plot figure too.  Defaults to square configuration.
        main_label (bool): True to remove the axis names from individual plots and places them as titles for main figure axes.
        confidence (float): Level of confidence between 0 and 1 at which confidence intervals are displayed.
    """
    categories: list = df_data[by].unique() if isinstance(by, str) else ['']
    num_plots: int = len(categories)*(len(df_data.columns)-1) if isinstance(by, str) else len(df_data.columns)
    ncol_fig: int = ncol_fig if isinstance(ncol_fig, int) else ceil(sqrt(num_plots))
    nrow_fig: int = ceil(num_plots / ncol_fig)
    fig, axes = plt.subplots(nrows=nrow_fig, ncols=ncol_fig, figsize=(5*ncol_fig, 5*nrow_fig))
    axes = axes.flatten() 
    axes_index: int = 0   

    for name in categories:
        plot_data: pd.DataFrame = df_data.query(f"{by} == '{name}'").drop(columns=[by]) if isinstance(by, str) else df_data

        for col in plot_data.columns:
            statistic, p_value = shapiro(plot_data[col])
            shapiro_info: str = f"    Shapiro-Wilk\n" +\
                                f"Statistic:  {statistic:.5f}\n"+\
                                f"P-Value:   {p_value:.5f}"
            qqplot(plot_data[col], ax=axes[axes_index], confidence=confidence)
            axes[axes_index].set_title(f'{name}', fontsize=10)
            axes[axes_index].text(0.02, 0.98, shapiro_info, transform=axes[axes_index].transAxes, fontsize=9, ha='left', va='top')
            axes_index += 1

    [fig.delaxes(ax) for ax in axes[-ncol_fig:] if not ax.has_data()]

    if main_label:
        fig.suptitle(f"Quantile-Quantile Plots for {''.join(df_data.columns.difference(['Species']))}", fontsize=10)
        fig.supxlabel("Theoretical quantiles", fontsize=10)
        fig.supylabel("Ordered quantiles", fontsize=10)
        [ax.set(xlabel='', ylabel='') for ax in axes]
    else:
        fig.suptitle(*df_data.columns.difference(['Species']))
    
    return fig
