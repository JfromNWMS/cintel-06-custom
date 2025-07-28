import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny.express import ui, input, render
from shiny import reactive
from pingouin import homoscedasticity
from stats_jordan import is_norm_small

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

ui.page_opts(title='Iris Dataset', fillable=True)

with ui.sidebar(title='Iris Dataset'):
    ui.input_select(
        'selected_variable',
        'Variable:',
        choices = continuous_variables,
        selected = continuous_variables[0]
    )

with ui.layout_columns():

    with ui.card(full_screen=True):
        ui.card_header('Summary Statistics of Full Dataset')

        @render.data_frame
        def datagrid_one():
            return render.DataGrid(format_describe(iris_df.describe()).reset_index())

    with ui.card(full_screen=True):
        ui.card_header("Boxplot and Summary Statistics")

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
                fontdict = {'fontsize': 9}
            )
            axes[1].axis('off')
            axes[1].set_title(input.selected_variable(), y=0.83, fontfamily='sans-serif', fontsize=11)
            fig.tight_layout(w_pad=3)
            return fig

with ui.layout_columns():

    with ui.card(full_screen=True):
        ui.card_header('Normality')

        @render.plot
        def normality():
            return is_norm_small(df_data=filtered_data(), by='Species', ncol_fig=3, main_label=False)
        
    with ui.card(full_screen=True):
        ui.card_header('Homoscedasticity and ANOVA')
        
        @render.data_frame
        def datagrid_two():
            levene_results = homoscedasticity(filtered_data(), dv=input.selected_variable(), group='Species')
            return render.DataGrid(levene_results)

@reactive.calc
def filtered_data():
    return iris_df[['Species', input.selected_variable()]]   