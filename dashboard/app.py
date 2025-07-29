import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import qqplot
from scipy.stats import shapiro
from math import ceil, sqrt
from shiny.express import ui, input, render
from shiny import reactive
from faicons import icon_svg

iris_df = sns.load_dataset('iris')
iris_df.columns = [col_name.replace('_',' ').title() for col_name in iris_df.columns] 
iris_df = iris_df.apply(lambda col: col.apply(lambda x: x.replace('_',' ').title()) if col.dtype == 'object' else col)
continuous_variables = [name for name in iris_df.columns if name != 'Species']

format_dict: dict = {
    'count': '{:.0f}'.format,
    'min': '{:.1f}'.format,
    'max': '{:.1f}'.format,
    'std': '{:.3f}'.format,
    'default': '{:.2f}'.format
}

def format_describe(df_data: pd.DataFrame) -> pd.DataFrame:
    return df_data.apply(lambda row: row.map(format_dict.get(row.name, format_dict['default'])), axis=1)

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

ui.page_opts(title='Iris Dashboard By Jordan', fillable=True)

with ui.sidebar(title='Iris Dataset'):

    with ui.value_box(
        showcase=icon_svg("leaf"),
        theme="bg-gradient-green-teal",
    ):
        "Unique Iris Species"

        @render.text
        def display_unique():
            return len(iris_df['Species'].unique())
        
    ui.input_select(
        'selected_variable',
        'Select Variable:',
        choices = continuous_variables,
        selected = continuous_variables[0]
    )

    ui.hr()
    ui.a("GitHub", href="https://github.com/JfromNWMS/cintel-06-custom", target="_blank")

with ui.layout_columns(max_height="50%"):

    with ui.card(full_screen=True):
        ui.card_header('Full Dataset')

        @render.data_frame
        def datagrid_one():
            return render.DataGrid(iris_df)
        
    with ui.card(full_screen=True):
        ui.card_header('Summary Statistics of Full Dataset')

        @render.data_frame
        def datagrid_two():
            df_summary = format_describe(iris_df.describe()).reset_index()
            df_summary.rename(columns={'index': ' '}, inplace=True)
            return render.DataGrid(df_summary)


with ui.layout_columns():

    with ui.card(full_screen=True):
        ui.card_header(f'Normality By Species')

        @render.plot
        def normality():
            return is_norm_small(df_data=filtered_data(), by='Species', ncol_fig=3, main_label=True)

    with ui.card(full_screen=True):
        ui.card_header("Boxplot and Summary Statistics By Species")

        @render.plot
        def box_summary():
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
            axes = axes.flatten()
            sns.boxplot(
                x='Species', y=input.selected_variable(), data=iris_df, hue='Species', ax=axes[0], showmeans=True, 
                meanprops={'marker':'o', 'markerfacecolor':'lightslategrey', 'markeredgecolor':'darkslategrey'}
            )
            plot_df = iris_df[['Species', input.selected_variable()]].groupby('Species').describe().T
            plot_df.index = plot_df.index.droplevel(0)
            plot_df = format_describe(plot_df)
            axes[1].text(
                0.5, 0.4, 
                plot_df.to_string().replace('ca', 'ca\n').replace('es', 'es:'),
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontfamily = 'monospace',
                fontdict = {'fontsize': 10}
            )
            axes[1].axis('off')
            axes[1].set_title(input.selected_variable(), y=0.83, fontfamily='sans-serif', fontsize=11)
            fig.tight_layout(w_pad=3)
            return fig

@reactive.calc
def filtered_data():
    return iris_df[['Species', input.selected_variable()]]   