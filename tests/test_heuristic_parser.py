from pathlib import Path
from multimodal_hybrid_parsing import DocumentParser


def test_pdf_parsing():
    # Create a parser instance
    parser = DocumentParser()

    # Get all files from samples directory
    samples_dir = Path("tests/samples")
    output_dir = Path("tests/output/markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

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
            # Load and convert the document
            parser.load_document(sample_file)

            # Create output path with same name but .md extension
            output_path = output_dir / f"{sample_file.stem}.md"

            # Convert to markdown
            markdown_content = parser.to_markdown(output_path)

            print(f"✓ Successfully converted {sample_file.name}")
            print(f"  Output saved to {output_path}")
            print(f"  Preview (first 200 chars):")
            print(f"  {markdown_content[:200]}...")
            print("-" * 50)

        except Exception as e:
            print(f"✗ Failed to convert {sample_file.name}")
            print(f"  Error: {str(e)}")
            print("-" * 50)


if __name__ == "__main__":
    test_pdf_parsing()
