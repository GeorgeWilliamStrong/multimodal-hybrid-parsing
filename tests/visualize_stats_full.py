import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the stats files
with open('tests/stats/heuristic_parsing_cuda_stats.json', 'r') as f:
    cuda_stats = json.load(f)

with open('tests/stats/heuristic_parsing_cpu_stats.json', 'r') as f:
    cpu_stats = json.load(f)

with open('tests/stats/heuristic_parsing_cuda_img_desc_stats.json', 'r') as f:
    cuda_img_desc_stats = json.load(f)

with open('tests/stats/pdf_plumber_stats.json', 'r') as f:
    pdf_plumber_stats = json.load(f)

with open('tests/stats/instill_hybrid_pipeline_stats.json', 'r') as f:
    instill_stats = json.load(f)

# Process data into pandas DataFrames
def process_stats(stats_dict, version):
    data = []
    for filename, entries in stats_dict.items():
        for entry in entries:
            # Clean filename and cap length
            clean_name = filename.split(']')[1].strip().replace('.pdf', '')
            if len(clean_name) > 20:  # Cap at 20 characters
                clean_name = clean_name[:17] + '...'
            
            row = {
                'filename': clean_name,
                'processing_time': entry['processing_time'],
                'file_size_mb': entry['file_size'] / (1024 * 1024),
                'version': version,
                'peak_memory': None,  # Default to None for memory metrics
                'memory_increase': None
            }
            
            # Add memory metrics if available
            if 'memory_metrics' in entry:
                metrics = entry['memory_metrics']
                if 'CUDA' in version:
                    row['peak_memory'] = metrics['peak_gpu_memory_mb']
                    row['memory_increase'] = metrics['gpu_memory_increase_mb']
                else:
                    row['peak_memory'] = metrics['final_cpu_memory_mb']
                    row['memory_increase'] = metrics['cpu_memory_increase_mb']
            
            data.append(row)
    return pd.DataFrame(data)

# Create DataFrames
cuda_df = process_stats(cuda_stats, 'Docling CUDA')
cpu_df = process_stats(cpu_stats, 'Docling CPU')
cuda_img_desc_df = process_stats(cuda_img_desc_stats, 'Docling CUDA+ImgDesc')
pdf_plumber_df = process_stats(pdf_plumber_stats, 'PDF Plumber')
instill_df = process_stats(instill_stats, 'Instill Hybrid')

# Combine all DataFrames
combined_df = pd.concat([
    cuda_df, cpu_df, cuda_img_desc_df, pdf_plumber_df, instill_df
])

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Create subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(14, 8))

# Processing Time Comparison
plot = sns.barplot(
    data=combined_df,
    x='filename',
    y='processing_time',
    hue='version',
    ax=ax1
)

# Customize plot
title = 'Processing Time Comparison Across All Methods'
ax1.set_title(title, pad=20, fontsize=14)
ax1.set_xlabel('Document', fontsize=12)
ax1.set_ylabel('Processing Time (seconds)', fontsize=12)
ax1.tick_params(axis='x', rotation=90)  # Change to vertical labels

# Adjust legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(
    'tests/stats/performance_comparison_full.png',
    dpi=300,
    bbox_inches='tight'
)

# Print some summary statistics
print("\nSummary Statistics (excluding WeWork S1):")
print("-" * 50)
versions = ['Docling CUDA', 'Docling CPU', 'Docling CUDA+ImgDesc', 
            'PDF Plumber', 'Instill Hybrid']

for version in versions:
    df = combined_df[
        (combined_df['version'] == version) & 
        (~combined_df['filename'].str.contains('wework-s1', case=False))
    ]
    avg_time = df['processing_time'].mean()
    max_time = df['processing_time'].max()
    
    print(f"\n{version} Version:")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Max processing time: {max_time:.2f} seconds")
    
    # Only print memory stats if available
    if df['peak_memory'].notna().any():
        avg_mem = df['peak_memory'].mean()
        max_mem = df['peak_memory'].max()
        print(f"Average peak memory: {avg_mem:.2f} MB")
        print(f"Max peak memory: {max_mem:.2f} MB")