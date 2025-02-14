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

# Process data into pandas DataFrames
def process_stats(stats_dict, version):
    data = []
    for filename, entries in stats_dict.items():
        for entry in entries:
            # Clean filename by removing tags and extension
            clean_name = filename.split(']')[1].strip().replace('.pdf', '')
            
            row = {
                'filename': clean_name,
                'processing_time': entry['processing_time'],
                'file_size_mb': entry['file_size'] / (1024 * 1024),
                'version': version
            }
            
            # Add memory metrics based on version type
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
cuda_df = process_stats(cuda_stats, 'CUDA')
cpu_df = process_stats(cpu_stats, 'CPU')
cuda_img_desc_df = process_stats(cuda_img_desc_stats, 'CUDA+ImgDesc')
combined_df = pd.concat([cuda_df, cpu_df, cuda_img_desc_df])

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Create subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(14, 8))

# Processing Time Comparison
sns.barplot(
    data=combined_df,
    x='filename',
    y='processing_time',
    hue='version',
    ax=ax1
)
title = 'Processing Time Comparison: CUDA vs CPU vs CUDA+ImgDesc'
ax1.set_title(title)
ax1.set_xlabel('Document')
ax1.set_ylabel('Processing Time (seconds)')
ax1.tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('tests/stats/performance_comparison.png', dpi=300, bbox_inches='tight')

# Print some summary statistics
print("\nSummary Statistics:")
print("-" * 50)
versions = ['CUDA', 'CPU', 'CUDA+ImgDesc']
for version in versions:
    df = combined_df[combined_df['version'] == version]
    print(f"\n{version} Version:")
    print(f"Average processing time: {df['processing_time'].mean():.2f} seconds")
    print(f"Max processing time: {df['processing_time'].max():.2f} seconds")
    print(f"Average peak memory: {df['peak_memory'].mean():.2f} MB")
    print(f"Max peak memory: {df['peak_memory'].max():.2f} MB") 