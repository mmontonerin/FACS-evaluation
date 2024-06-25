import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Read the CSV file into a DataFrame
df = pd.read_csv("all_sort_files_fix_with_log_and_pcr_binary.csv")


# Define the PCR columns to exclude
pcr_exclude_cols = ['pcr_cn', 'pcr_cm', 'pcr_cf', 'pcr_cb']

# Filter out samples that have TRUE in any of the pcr_exclude_cols
df_filtered = df[~df[pcr_exclude_cols].any(axis=1)]

# Define the columns you want to plot
columns_to_plot = ['488-530/40-Height', '488-SSC-Height', '488-FSC1-Height', '488-530/40-Area']

# Define colors based on the 'pcr' column
colors = {'Acaulosporaceae': '#7da7d9', 'Glomeraceae': '#a481b5', 'Diversisporaceae': '#a3d39c', 'Gigasporaceae': '#39b54a', 'Clareidoglomeraceae': '#fff200', 'Ambisporaceae': '#f7941d', 'Paraglomeraceae': '#ed1c24'}

# Define markers and labels for PCR columns
pcr_markers = {'pcr_f': 'o', 'pcr_b': '^', 'pcr_n': 'X', 'pcr_m': 'd'}
pcr_labels = {'pcr_f': 'fungi', 'pcr_b': 'bacteria', 'pcr_n': 'negative', 'pcr_m': 'mixed'}

# Create a folder to save the plots
os.makedirs("plots_alltogether", exist_ok=True)

# Function to create valid filenames
def make_valid_filename(column_x, column_y):
    return f"all_families_scatter_{column_x.replace('/', '_')}_vs_{column_y.replace('/', '_')}"

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

# Plot all families together log
for column_x, column_y in [('488-SSC-Height-Log', '488-530/40-Height-Log'), 
                           ('488-FSC1-Height-Log', '488-530/40-Height-Log'), 
                           ('488-SSC-Height-Log', '488-530/40-Area-Log')]:
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for family, color in colors.items():
        family_group = df_filtered[df_filtered['Family'] == family]
        for pcr, marker in pcr_markers.items():
            pcr_group = family_group[family_group[pcr] == True]
            if not pcr_group.empty:
                plt.scatter(pcr_group[column_x], pcr_group[column_y], label=f'{family} ({pcr})', color=color, marker=marker, edgecolors='k', linewidths=0.5)
    
    plt.xlabel(column_x)
    plt.ylabel(column_y)
    plt.title(f'{column_x} vs {column_y}')
 
    # Create legend for families
    handles_families = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=family) for family, color in colors.items()]
    legend_families = plt.legend(handles=handles_families, title='Families', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    # Calculate overall statistics
    mean_x = df_filtered[column_x].mean()
    std_x = df_filtered[column_x].std()
    mean_y = df_filtered[column_y].mean()
    std_y = df_filtered[column_y].std()

    legend_handles = []

    for pcr, marker in pcr_markers.items():
        pcr_row = df_filtered[df_filtered[pcr] == True]

        # Plot confidence ellipses
        if pcr == 'pcr_f':
            plot_confidence_ellipse(pcr_row[column_x], pcr_row[column_y], ax, color='black')
        elif pcr == 'pcr_b':
            plot_confidence_ellipse(pcr_row[column_x], pcr_row[column_y], ax, color='grey', linestyle='--')
        elif pcr == 'pcr_n':
            plot_confidence_ellipse(pcr_row[column_x], pcr_row[column_y], ax, color='grey')
        elif pcr == 'pcr_m':
            plot_confidence_ellipse(pcr_row[column_x], pcr_row[column_y], ax, color='black', linestyle='--')


        if len(pcr_row) >= 2:
            r, p_value = pearsonr(pcr_row[column_x], pcr_row[column_y])
            plt.annotate(f'{pcr_labels[pcr]}: r={r:.2f}, p={p_value:.2g}', 
                        xy=(0.01, 0.99 - 0.05 * list(pcr_markers.keys()).index(pcr)), xycoords='axes fraction', fontsize=12, verticalalignment='top')
        
        xlab = column_x
        ylab = column_y

        plt.xlabel(f'{xlab}\nMean={mean_x:.2f}, SD={std_x:.2f}')
        plt.ylabel(f'{ylab}\nMean={mean_y:.2f}, SD={std_y:.2f}')        
        plt.title(f'All families')


    # Create legend for PCR markers
    handles_pcr = [plt.Line2D([0], [0], marker=marker, color='k', linestyle='None', markersize=10, label=pcr) for pcr, marker in pcr_markers.items()]
    legend_pcr = plt.legend(handles=handles_pcr, title='PCR Markers', bbox_to_anchor=(1.05, 0.5), loc='upper left')

    # Statistical significance annotation fungi vs bacteria
    if 'pcr_f' in df_filtered.columns and 'pcr_b' in df_filtered.columns:
        pcr_f_values = df_filtered[df_filtered['pcr_f'] == True][[column_x, column_y]].values.flatten()
        pcr_b_values = df_filtered[df_filtered['pcr_b'] == True][[column_x, column_y]].values.flatten()

        if len(pcr_f_values) > 0 and len(pcr_b_values) > 0:
            t_stat_b, p_val_b = ttest_ind(pcr_f_values, pcr_b_values)
            plt.annotate(f'fungi vs bacteria: p={p_val_b:.2g}', 
                        xy=(0.01, 0.05), xycoords='axes fraction', fontsize=12, verticalalignment='top')

    # Add the number of events to the plot
    n_events = df_filtered.shape[0]
    plt.text(0.99, 0.01, f'N = {n_events}', verticalalignment='bottom', horizontalalignment='right',
        transform=plt.gca().transAxes, fontsize=12, color='black')

    plt.gca().add_artist(legend_families)

    valid_filename = make_valid_filename(column_x, column_y)
    plt.savefig(f"plots_alltogether/all_fam_norm_{valid_filename}.pdf", bbox_inches='tight', bbox_extra_artists=[legend_families, legend_pcr])
    plt.savefig(f"plots_alltogether/all_fam_norm_{valid_filename}.png", bbox_inches='tight', bbox_extra_artists=[legend_families, legend_pcr])
    plt.close()






