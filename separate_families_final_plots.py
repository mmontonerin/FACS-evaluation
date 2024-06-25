import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import pearsonr, ttest_ind

# Read the CSV file into a DataFrame
df = pd.read_csv("all_sort_files_fix_with_log_and_pcr_binary.csv")

# Define the PCR columns to exclude
pcr_exclude_cols = ['pcr_cn', 'pcr_cm', 'pcr_cf', 'pcr_cb']

# Filter out samples that have TRUE in any of the pcr_exclude_cols
df_filtered = df[~df[pcr_exclude_cols].any(axis=1)]

# Define columns to plot
columns_to_plot = ['488-530/40-Height', '488-SSC-Height', '488-FSC1-Height', '488-530/40-Area']

# Define colors based on the 'pcr' column
colors = {'Acaulosporaceae': '#7da7d9', 'Glomeraceae': '#a481b5', 'Diversisporaceae': '#a3d39c',
          'Gigasporaceae': '#39b54a', 'Clareidoglomeraceae': '#fff200', 'Ambisporaceae': '#f7941d',
          'Paraglomeraceae': '#ed1c24', 'Archaeosporaceae': 'black'}

# Define markers and labels for PCR columns
pcr_markers = {'pcr_f': 'o', 'pcr_b': '^', 'pcr_n': 'X', 'pcr_m': 'd'}
pcr_labels = {'pcr_f': 'fungi', 'pcr_b': 'bacteria', 'pcr_n': 'negative', 'pcr_m': 'mixed'}

# Create a folder to save the plots
os.makedirs("plots_separatefamilies", exist_ok=True)

# Log normalization function
def log_normalize(df, columns):
    for col in columns:
        df[col + '-Log-sep'] = np.log(df[col])
    return df

# Function to create valid filenames
def make_valid_filename(column_x, column_y):
    return f"{column_x.replace('/', '_')}_vs_{column_y.replace('/', '_')}"

def clean_column_name(column_name):
    return column_name.replace('-sep', '')

# Function to plot confidence ellipse
def plot_confidence_ellipse(x, y, ax, n_std=1.0, color='black', linestyle='solid', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, fill=False, edgecolor=color, linestyle=linestyle, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Perform log normalization for each family separately
for family in colors:
    family_group = df_filtered[df_filtered['Family'] == family].copy()
    family_group = log_normalize(family_group, columns_to_plot)

    # Plot each combination of columns
    for column_x, column_y in [('488-SSC-Height-Log-sep', '488-530/40-Height-Log-sep'),
                               ('488-FSC1-Height-Log-sep', '488-530/40-Height-Log-sep'),
                               ('488-SSC-Height-Log-sep', '488-530/40-Area-Log-sep')]:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Calculate overall statistics
        mean_x = family_group[column_x].mean()
        std_x = family_group[column_x].std()
        mean_y = family_group[column_y].mean()
        std_y = family_group[column_y].std()

        legend_handles = []

        for pcr, marker in pcr_markers.items():
            pcr_group = family_group[family_group[pcr] == True]
            if not pcr_group.empty:
                scatter = plt.scatter(pcr_group[column_x], pcr_group[column_y], label=f'{family} ({pcr_labels[pcr]})',
                                      color=colors[family], marker=marker, edgecolors='k', linewidths=0.5)
                legend_handles.append(scatter)

                # Plot confidence ellipses
                if pcr == 'pcr_f':
                    plot_confidence_ellipse(pcr_group[column_x], pcr_group[column_y], ax, color='black')
                elif pcr == 'pcr_b':
                    plot_confidence_ellipse(pcr_group[column_x], pcr_group[column_y], ax, color='grey', linestyle='--')
                elif pcr == 'pcr_n':
                    plot_confidence_ellipse(pcr_group[column_x], pcr_group[column_y], ax, color='grey')
                elif pcr == 'pcr_m':
                    plot_confidence_ellipse(pcr_group[column_x], pcr_group[column_y], ax, color='black', linestyle='--')

                # Pearson correlation coefficient
                r, p_value = pearsonr(pcr_group[column_x], pcr_group[column_y])
                plt.annotate(f'{pcr_labels[pcr]}: r={r:.2f}, p={p_value:.2g}', 
                             xy=(0.01, 0.99 - 0.05 * list(pcr_markers.keys()).index(pcr)), xycoords='axes fraction', fontsize=12, verticalalignment='top')
        
        xlab = clean_column_name(column_x)
        ylab = clean_column_name(column_y)

        plt.xlabel(f'{xlab}\nMean={mean_x:.2f}, SD={std_x:.2f}')
        plt.ylabel(f'{ylab}\nMean={mean_y:.2f}, SD={std_y:.2f}')        
        plt.title(f'{family}')

        # Create legend for PCR markers
        handles_pcr = [plt.Line2D([0], [0], marker=marker, color='k', linestyle='None', markersize=10, label=pcr_labels[pcr]) for pcr, marker in pcr_markers.items()]
        legend_pcr = plt.legend(handles=handles_pcr, title='PCR Markers', bbox_to_anchor=(1.05, 0.5), loc='upper left')
        plt.gca().add_artist(legend_pcr)

        # Statistical significance annotation
        if 'pcr_f' in family_group.columns and 'pcr_n' in family_group.columns and 'pcr_b' in family_group.columns:
            pcr_f_values = family_group[family_group['pcr_f'] == True][[column_x, column_y]].values.flatten()
            pcr_n_values = family_group[family_group['pcr_n'] == True][[column_x, column_y]].values.flatten()
            pcr_b_values = family_group[family_group['pcr_b'] == True][[column_x, column_y]].values.flatten()

            if len(pcr_f_values) > 0 and len(pcr_n_values) > 0:
                t_stat_n, p_val_n = ttest_ind(pcr_f_values, pcr_n_values)
                plt.annotate(f'fungi vs negative: p={p_val_n:.2g}', 
                             xy=(0.01, 0.1), xycoords='axes fraction', fontsize=12, verticalalignment='top')

            if len(pcr_f_values) > 0 and len(pcr_b_values) > 0:
                t_stat_b, p_val_b = ttest_ind(pcr_f_values, pcr_b_values)
                plt.annotate(f'fungi vs bacteria: p={p_val_b:.2g}', 
                             xy=(0.01, 0.05), xycoords='axes fraction', fontsize=12, verticalalignment='top')

        # Add the number of events to the plot
        n_events = family_group.shape[0]
        plt.text(0.99, 0.01, f'N = {n_events}', verticalalignment='bottom', horizontalalignment='right',
            transform=plt.gca().transAxes, fontsize=12, color='black')


        valid_filename = make_valid_filename(column_x, column_y)
        plt.savefig(f"plots_separatefamilies/{family}_{valid_filename}.pdf", bbox_inches='tight', bbox_extra_artists=[legend_pcr])
        plt.savefig(f"plots_separatefamilies/{family}_{valid_filename}.png", bbox_inches='tight', bbox_extra_artists=[legend_pcr])
        plt.close()

    # Standardize the data
    x = family_group[columns_to_plot].values
    x = StandardScaler().fit_transform(x)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Add the Family and PCR columns back to the DataFrame
    principal_df = pd.concat([principal_df, family_group[['Family'] + list(pcr_markers.keys())].reset_index(drop=True)], axis=1)

    # Create a PCA plot
    plt.figure(figsize=(14, 10))
    for pcr, marker in pcr_markers.items():
        pcr_group = principal_df[principal_df[pcr] == True]
        if not pcr_group.empty:
            plt.scatter(pcr_group['PC1'], pcr_group['PC2'], label=f'{family} ({pcr_labels[pcr]})', color=colors[family], marker=marker, edgecolors='k', linewidths=0.5)

            # Calculate average position of samples with this PCR marker
            avg_PC1 = pcr_group['PC1'].mean()
            avg_PC2 = pcr_group['PC2'].mean()

            # Plot arrow from origin to average position
            plt.arrow(0, 0, avg_PC1, avg_PC2, color='black', alpha=0.8, linewidth=1.5, linestyle='--')
            plt.text(avg_PC1 * 1.1, avg_PC2 * 1.1, f"{pcr_labels[pcr]}", fontsize=12, color='black', alpha=1)

    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
    plt.title(f'PCA of coordinates - {family}')

    # Create legend for PCR markers
    handles_pcr = [plt.Line2D([0], [0], marker=marker, color='k', linestyle='None', markersize=10, label=pcr_labels[pcr]) for pcr, marker in pcr_markers.items()]
    legend_pcr = plt.legend(handles=handles_pcr, title='PCR Markers', bbox_to_anchor=(1.05, 0.5), loc='upper left')
    plt.gca().add_artist(legend_pcr)

    # Add the number of events to the plot
    n_events = family_group.shape[0]
    plt.text(0.99, 0.01, f'N = {n_events}', verticalalignment='bottom', horizontalalignment='right',
        transform=plt.gca().transAxes, fontsize=12, color='black')

    plt.savefig(f"plots_separatefamilies/{family}_pca_plot.pdf", bbox_inches='tight', bbox_extra_artists=[legend_pcr])
    plt.savefig(f"plots_separatefamilies/{family}_pca_plot.png", bbox_inches='tight', bbox_extra_artists=[legend_pcr])
    plt.close()

    # Select columns for correlation analysis
    selected_columns2 = ['pcr_n', 'pcr_m', 'pcr_f', 'pcr_b','488-530/40-Height-Log-sep','488-SSC-Height-Log-sep','488-FSC1-Height-Log-sep','488-530/40-Area-Log-sep']

    # Correlation analysis
    correlation_matrix = family_group[selected_columns2].corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'{family}: Correlation Matrix of Selected Features')
    plt.savefig(f"plots_separatefamilies/{family}_correlation_matrix.pdf")
    plt.savefig(f"plots_separatefamilies/{family}_correlation_matrix.png")
    plt.close()

    # Visualize pairwise relationships
    pairplot = sns.pairplot(family_group[selected_columns2])
    pairplot.savefig(f'plots_separatefamilies/{family}_pairwise_relationships.pdf')
    pairplot.savefig(f'plots_separatefamilies/{family}_pairwise_relationships.png')
    plt.close()
