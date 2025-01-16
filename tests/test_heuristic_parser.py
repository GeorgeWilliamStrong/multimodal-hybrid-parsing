import time
import json
from pathlib import Path
from multimodal_hybrid_parsing import DocumentParser


def test_pdf_parsing():
    # Create a parser instance
    parser = DocumentParser()

    # Get all files from samples directory
    samples_dir = Path("tests/samples")
    output_dir = Path("tests/output/markdown/heuristic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timing stats directory
    stats_dir = Path("tests/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    timing_file = stats_dir / "heuristic_parsing_times.json"

    # Load existing timing data if available
    timing_data = {}
    if timing_file.exists():
        timing_data = json.loads(timing_file.read_text())

    # Get all PDF files
    supported_extensions = [".pdf"]
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

            # Load and convert the document
            parser.load_document(sample_file)

            # Create output path with same name but .md extension
            output_path = output_dir / f"{sample_file.stem}.md"

            # Convert to markdown
            markdown_content = parser.to_markdown(output_path)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update timing data
            if sample_file.name not in timing_data:
                timing_data[sample_file.name] = []
            timing_data[sample_file.name].append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': processing_time,
                'file_size': sample_file.stat().st_size,
                'success': True
            })

            print(f"✓ Successfully converted {sample_file.name}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Output saved to {output_path}")
            print(f"  Preview (first 200 chars):")
            print(f"  {markdown_content[:200]}...")
            print("-" * 50)

        except Exception as e:
            # Record failed attempt
            if sample_file.name not in timing_data:
                timing_data[sample_file.name] = []
            timing_data[sample_file.name].append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'file_size': sample_file.stat().st_size,
                'success': False
            })

            print(f"✗ Failed to convert {sample_file.name}")
            print(f"  Error: {str(e)}")
            print("-" * 50)

    # Save timing data
    timing_file.write_text(json.dumps(timing_data, indent=2))


if __name__ == "__main__":
    test_pdf_parsing()
