import time
import json
from pathlib import Path
import psutil
from multimodal_hybrid_parsing import DocumentParser


def test_pdf_parsing():
    # Create a parser instance
    parser = DocumentParser(device="cpu")

    # Get all files from samples directory
    samples_dir = Path("tests/samples")
    output_dir = Path("tests/output/markdown/heuristic_cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timing stats directory
    stats_dir = Path("tests/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = stats_dir / "heuristic_parsing_cpu_stats.json"

    # Load existing timing data if available
    benchmark_data = {}
    if benchmark_file.exists():
        benchmark_data = json.loads(benchmark_file.read_text())

    # Get all PDF files
    supported_extensions = [".pdf", ".docx", ".pptx"]
    sample_files = [
        f for f in samples_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not sample_files:
        print(f"No supported documents found in {samples_dir}")
        return

    print(f"Found {len(sample_files)} documents to process")
    print("-" * 50)

    # Process each file
    for sample_file in sample_files:
        try:
            start_time = time.time()

            # Record initial memory state
            initial_cpu_memory = (
                psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            )

            # Process the document
            parser.process_document(sample_file)

            # Create output path with same name but .md extension
            output_path = output_dir / f"{sample_file.stem}.md"

            # Get markdown content and save to file
            markdown_pages = parser.get_page_markdown()
            markdown_content = "\n\n".join(markdown_pages)
            output_path.write_text(markdown_content)

            # Calculate memory usage
            final_cpu_memory = (
                psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update benchmark data
            if sample_file.name not in benchmark_data:
                benchmark_data[sample_file.name] = []
            benchmark_data[sample_file.name].append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': processing_time,
                'file_size': sample_file.stat().st_size,
                'success': True,
                'memory_metrics': {
                    'initial_cpu_memory_mb': round(initial_cpu_memory, 2),
                    'final_cpu_memory_mb': round(final_cpu_memory, 2),
                    'cpu_memory_increase_mb': round(
                        final_cpu_memory - initial_cpu_memory, 2
                    )
                }
            })

            print(f"✓ Successfully converted {sample_file.name}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(
                f"  CPU Memory: "
                f"{final_cpu_memory - initial_cpu_memory:.2f}MB increase"
            )
            print(f"  Output saved to {output_path}")
            print("  Preview (first 100 chars):")
            print(f"  {markdown_content[:100]}...")
            print("-" * 50)

        except Exception as e:
            # Record failed attempt
            if sample_file.name not in benchmark_data:
                benchmark_data[sample_file.name] = []
            benchmark_data[sample_file.name].append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'file_size': sample_file.stat().st_size,
                'success': False
            })

            print(f"✗ Failed to convert {sample_file.name}")
            print(f"  Error: {str(e)}")
            print("-" * 50)

    # Save benchmark data
    benchmark_file.write_text(json.dumps(benchmark_data, indent=2))


if __name__ == "__main__":
    test_pdf_parsing()
