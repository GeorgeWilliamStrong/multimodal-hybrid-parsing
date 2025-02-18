import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from pathlib import Path


def get_pdf_page_count(pdf_path):
    """Get the number of pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


def process_stats_with_pages(stats_dict, version, samples_dir):
    """Process stats and add page counts."""
    data = []
    for filename, entries in stats_dict.items():
        # Get PDF path
        pdf_path = samples_dir / filename
        
        # Get page count
        n_pages = get_pdf_page_count(pdf_path)
        if n_pages is None:
            continue
            
        for entry in entries:
            # Determine group
            img_desc_methods = [
                'Docling CUDA+Granite',
                'Docling CUDA+SmolVLM',
                'Instill Hybrid'
            ]
            if version in img_desc_methods:
                group = 'With Image Description'
            else:
                group = 'Without Image Description'
                
            row = {
                'filename': filename.split(']')[1].strip().replace('.pdf', ''),
                # Convert seconds to minutes
                'processing_time': entry['processing_time'] / 60.0,
                'file_size_mb': entry['file_size'] / (1024 * 1024),
                'version': version,
                'pages': n_pages,
                'group': group
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


def main():
    # Setup paths
    samples_dir = Path('tests/samples')
    stats_dir = Path('tests/stats')
    
    # Load stats files
    stats_files = {
        'Docling CUDA': 'heuristic_parsing_cuda_stats.json',
        'Docling CPU': 'heuristic_parsing_cpu_stats.json',
        'Docling CUDA+Granite': (
            'heuristic_parsing_cuda_img_desc_granite_stats.json'
        ),
        'Docling CUDA+SmolVLM': (
            'heuristic_parsing_cuda_img_desc_smolVlm_stats.json'
        ),
        'PDF Plumber': 'pdf_plumber_stats.json',
        'Instill Hybrid': 'instill_hybrid_pipeline_stats.json'
    }
    
    # Process all stats files
    dataframes = []
    for version, filename in stats_files.items():
        with open(stats_dir / filename, 'r') as f:
            stats = json.load(f)
            df = process_stats_with_pages(stats, version, samples_dir)
            dataframes.append(df)
    
    # Combine all data
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Print page counts and file sizes for each file
    print("\nFile Statistics:")
    print("-" * 70)
    headers = f"{'Filename':<35} {'Pages':>8} {'Size (MB)':>12}"
    print(headers)
    print("-" * 70)
    unique_files = combined_df[
        ['filename', 'pages', 'file_size_mb']
    ].drop_duplicates()
    for _, row in unique_files.sort_values('filename').iterrows():
        print(
            f"{row['filename'][:35]:<35} {row['pages']:>8d}"
            f" {row['file_size_mb']:>12.2f}"
        )
    print("-" * 70 + "\n")
    
    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    sns.set_style("whitegrid")
    
    # Filter data for each group
    img_desc = combined_df[combined_df['group'] == 'With Image Description']
    no_img = combined_df[combined_df['group'] == 'Without Image Description']
    
    # Plot methods with image description
    sns.scatterplot(
        data=img_desc,
        x='pages',
        y='processing_time',
        style='version',
        hue='version',
        markers=['o', 's', '^'],
        s=90,
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
        ax=ax1
    )
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_title('With Image Description', pad=20, fontsize=14)
    ax1.set_xlabel('Pages (log)', fontsize=12)
    ax1.set_ylabel('Processing Time (minutes)', fontsize=12)
    
    # Plot methods without image description
    sns.scatterplot(
        data=no_img,
        x='pages',
        y='processing_time',
        style='version',
        hue='version',
        markers=['o', 's', '^'],
        s=90,
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
        ax=ax2
    )
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_title('Without Image Description', pad=20, fontsize=14)
    ax2.set_xlabel('Pages (log)', fontsize=12)
    ax2.set_ylabel('')  # Explicitly set empty y-axis label
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(
        'tests/stats/pages_vs_time.png',
        dpi=300,
        bbox_inches='tight'
    )
    
    # Print correlation statistics
    print("\nCorrelation between Pages and Processing Time:")
    print("-" * 50)
    for version in stats_files.keys():
        version_data = combined_df[combined_df['version'] == version]
        corr = version_data['pages'].corr(version_data['processing_time'])
        print(f"{version}: {corr:.3f}")


if __name__ == '__main__':
    main() 