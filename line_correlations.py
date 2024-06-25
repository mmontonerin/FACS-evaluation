import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# Read CSV file
df = pd.read_csv("all_sort_files_fix_with_log_and_pcr_binary.csv")

# Step 1: Remove rows where 'line' is empty or NaN
df_cleaned = df.dropna(subset=['line'])

# Convert 'line' column to boolean if necessary
if not isinstance(df_cleaned['line'].iloc[0], bool):
    df_cleaned.loc[:, 'line'] = df_cleaned['line'].apply(lambda x: x.strip().lower() == 'true')

# Ensure 'line' column is boolean type
df_cleaned['line'] = df_cleaned['line'].astype(bool)

# Print unique values in 'line' column to check for variations
print("Unique values in 'line' column:", df_cleaned['line'].unique())

# Step 3: Split into df_true_line and df_false_line based on 'line' value
df_true_line = df_cleaned[df_cleaned['line']]
df_false_line = df_cleaned[~df_cleaned['line']]

# Initialize dictionaries to store true counts
true_counts_true_line = {}
true_counts_false_line = {}
p_values = {}

# List of PCR columns
pcr_columns = ['pcr_b', 'pcr_f', 'pcr_m', 'pcr_n']

# Calculate true counts and perform chi-squared test for each PCR column
for pcr_col in pcr_columns:
    true_counts_true_line[pcr_col] = df_true_line[pcr_col].sum()
    true_counts_false_line[pcr_col] = df_false_line[pcr_col].sum()

    # Create contingency table
    contingency_table = [
        [true_counts_true_line[pcr_col], len(df_true_line) - true_counts_true_line[pcr_col]],
        [true_counts_false_line[pcr_col], len(df_false_line) - true_counts_false_line[pcr_col]]
    ]
    
    # Perform chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Store p-value
    p_values[pcr_col] = p

# Plot the bar chart
pcr_labels = {'pcr_f': 'fungi', 'pcr_b': 'bacteria', 'pcr_n': 'negative', 'pcr_m': 'mixed'}
labels = list(pcr_labels.keys())
label_names = [pcr_labels[col] for col in labels]
true_line_counts = [true_counts_true_line[col] for col in labels]
false_line_counts = [true_counts_false_line[col] for col in labels]
x = range(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))

# Define bar colors
colors_true_line = ['black'] * len(labels)  
colors_false_line = ['grey'] * len(labels)  

bar_width = 0.35
bar1 = ax.bar(x, true_line_counts, bar_width, label='Separate cluster', color=colors_true_line)
bar2 = ax.bar([p + bar_width for p in x], false_line_counts, bar_width, label='No separate cluster', color=colors_false_line)

# Add significance markers
alpha = 0.05
max_count = max(true_line_counts + false_line_counts)
y_offset = 0.05 * max_count  # Offset for asterisks above the tallest bar
for i, pcr_col in enumerate(labels):
    p_value = p_values[pcr_col]
    max_count = max(true_line_counts + false_line_counts)
    if p_value < 0.01:
        # Place the double asterisk above the taller bar
        if true_line_counts[i] > false_line_counts[i]:
            y = true_line_counts[i] + 0.05 * max_count
            ax.text(i, y, '**', ha='center', va='bottom', color='black')
        else:
            y = false_line_counts[i] + 0.05 * max_count
            ax.text(i + bar_width, y, '**', ha='center', va='bottom', color='black')
    elif p_value < alpha:
        # Place the single asterisk above the taller bar
        if true_line_counts[i] > false_line_counts[i]:
            y = true_line_counts[i] + 0.05 * max_count
            ax.text(i, y, '*', ha='center', va='bottom', color='black')
        else:
            y = false_line_counts[i] + 0.05 * max_count
            ax.text(i + bar_width, y, '*', ha='center', va='bottom', color='black')

# Add labels, title, and legend
ax.set_xlabel('PCR markers')
ax.set_ylabel('Positive PCR result')
ax.set_title('PCR results by FACS coordinates visualization')
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels(label_names)
ax.legend()

# Adjust y-axis limit
ax.set_ylim(0, max_count + 2 * y_offset)

# Adjust layout to prevent overlap
plt.xticks(rotation=45)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add more space for the title

# Save the plot
plt.savefig("plots_alltogether/line_pcr_comparison.png", bbox_inches='tight')
plt.savefig("plots_alltogether/line_pcr_comparison.pdf", bbox_inches='tight')

plt.close()