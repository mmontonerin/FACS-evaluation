import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Read the CSV file into a DataFrame
df = pd.read_csv("all_sort_files_fix_with_log_and_pcr_binary.csv")


# Define the PCR columns to exclude
pcr_exclude_cols = ['pcr_cn', 'pcr_cm', 'pcr_cf', 'pcr_cb']

# Filter out samples that have TRUE in any of the pcr_exclude_cols
df_filtered = df[~df[pcr_exclude_cols].any(axis=1)]

selected_columns = ['488-530/40-Height-Log','488-SSC-Height-Log','488-FSC1-Height-Log','488-530/40-Area-Log']

# Define colors based on the 'Family' column
colors = {'Acaulosporaceae': '#7da7d9', 'Glomeraceae': '#a481b5', 'Diversisporaceae': '#a3d39c', 'Gigasporaceae': '#39b54a', 'Clareidoglomeraceae': '#fff200', 'Ambisporaceae': '#f7941d', 'Paraglomeraceae': '#ed1c24'}

# Define markers and labels for PCR columns
pcr_markers = {'pcr_f': 'o', 'pcr_b': '^', 'pcr_n': 'X', 'pcr_m': 'd'}
pcr_labels = {'pcr_f': 'fungi', 'pcr_b': 'bacteria', 'pcr_n': 'negative', 'pcr_m': 'mixed'}

# Ensure data integrity and convert to numeric if necessary
#df_filtered[selected_columns] = df_filtered[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values if any
#df_filtered.dropna(subset=selected_columns, inplace=True)


# Standardize the data
x = df_filtered[selected_columns].values
x = StandardScaler().fit_transform(x)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the Family and PCR columns back to the DataFrame
principal_df = pd.concat([principal_df, df_filtered[['Family'] + list(pcr_markers.keys())].reset_index(drop=True)], axis=1)

# Calculate average positions for each PCR marker across all data points
avg_positions = {}
for pcr in pcr_markers:
    pcr_group = principal_df[principal_df[pcr] == True]
    avg_positions[pcr] = (pcr_group['PC1'].mean(), pcr_group['PC2'].mean())

# Create a PCA plot
plt.figure(figsize=(14, 10))
for family, color in colors.items():
    for pcr, marker in pcr_markers.items():
        pcr_group = principal_df[(principal_df['Family'] == family) & (principal_df[pcr] == True)]
        if not pcr_group.empty:
            plt.scatter(pcr_group['PC1'], pcr_group['PC2'], label=f'{family} ({pcr_labels[pcr]})', color=color,
                        marker=marker, edgecolors='k', linewidths=0.5)

# Plot arrows from the origin to the average positions of each PCR marker
for pcr, avg_pos in avg_positions.items():
    avg_PC1, avg_PC2 = avg_pos
    plt.arrow(0, 0, avg_PC1, avg_PC2, color='black', alpha=0.8, linewidth=1.5, linestyle='--')
    plt.text(avg_PC1 * 1.1, avg_PC2 * 1.1, f"{pcr_labels[pcr]}", fontsize=12, color='black', alpha=1)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of coordinates')
# Create legend for families
handles_families = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=family) for family, color in colors.items()]
legend_families = plt.legend(handles=handles_families, title='Families', bbox_to_anchor=(1.05, 1), loc='upper left')
    
# Create legend for PCR markers
handles_pcr = [plt.Line2D([0], [0], marker=marker, color='k', linestyle='None', markersize=10, label=pcr) for pcr, marker in pcr_markers.items()]
legend_pcr = plt.legend(handles=handles_pcr, title='PCR Markers', bbox_to_anchor=(1.05, 0.5), loc='upper left')

# Add the number of events to the plot
n_events = df_filtered.shape[0]
plt.text(0.99, 0.01, f'N = {n_events}', verticalalignment='bottom', horizontalalignment='right',
    transform=plt.gca().transAxes, fontsize=12, color='black')


plt.gca().add_artist(legend_families)
plt.savefig("plots_alltogether/pca_plot.pdf", bbox_inches='tight', bbox_extra_artists=[legend_families, legend_pcr])
plt.savefig("plots_alltogether/pca_plot.png", bbox_inches='tight', bbox_extra_artists=[legend_families, legend_pcr])
plt.close()



selected_columns2 = ['pcr_n', 'pcr_m', 'pcr_f', 'pcr_b','488-530/40-Height-Log','488-SSC-Height-Log','488-FSC1-Height-Log','488-530/40-Area-Log']


# Correlation analysis
correlation_matrix = df_filtered[selected_columns2].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Selected Features')
plt.savefig("plots_alltogether/correlation_matrix.pdf")
plt.savefig("plots_alltogether/correlation_matrix.png")
plt.close()

# Visualize pairwise relationships
pairplot = sns.pairplot(df_filtered[selected_columns2])
pairplot.savefig('plots_alltogether/pairwise_relationships.pdf')
pairplot.savefig('plots_alltogether/pairwise_relationships.png')
