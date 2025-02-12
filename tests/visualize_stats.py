import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the stats files
with open('tests/stats/heuristic_parsing_cuda_stats.json', 'r') as f:
    cuda_stats = json.load(f)

with open('tests/stats/heuristic_parsing_cpu_stats.json', 'r') as f:
    cpu_stats = json.load(f)

# Process data into pandas DataFrames
def process_stats(stats_dict, version):
    data = []
    for filename, entries in stats_dict.items():
        for entry in entries:
            row = {
                'filename': filename.split(']')[1].strip().replace('.pdf', ''),  # Clean filename
                'processing_time': entry['processing_time'],
                'file_size_mb': entry['file_size'] / (1024 * 1024),  # Convert to MB
                'version': version
            }
            
            # Add memory metrics
            if version == 'CUDA':
                row['peak_memory'] = entry['memory_metrics']['peak_gpu_memory_mb']
                row['memory_increase'] = entry['memory_metrics']['gpu_memory_increase_mb']
            else:
                row['peak_memory'] = entry['memory_metrics']['final_cpu_memory_mb']
                row['memory_increase'] = entry['memory_metrics']['cpu_memory_increase_mb']
            
            data.append(row)
    return pd.DataFrame(data)

# Create DataFrames
cuda_df = process_stats(cuda_stats, 'CUDA')
cpu_df = process_stats(cpu_stats, 'CPU')
combined_df = pd.concat([cuda_df, cpu_df])

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Create subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))

# 1. Processing Time Comparison
sns.barplot(data=combined_df, x='filename', y='processing_time', hue='version', ax=ax1)
ax1.set_title('Processing Time Comparison: CUDA vs CPU')
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
for version in ['CUDA', 'CPU']:
    df = combined_df[combined_df['version'] == version]
    print(f"\n{version} Version:")
    print(f"Average processing time: {df['processing_time'].mean():.2f} seconds")
    print(f"Max processing time: {df['processing_time'].max():.2f} seconds")
    print(f"Average peak memory: {df['peak_memory'].mean():.2f} MB")
    print(f"Max peak memory: {df['peak_memory'].max():.2f} MB") 