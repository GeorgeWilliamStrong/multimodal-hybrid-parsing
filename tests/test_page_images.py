from pathlib import Path
from multimodal_hybrid_parsing.page_images import PageImageConverter


def test_document_to_images():
    # Create a converter instance
    converter = PageImageConverter(dpi=200)

    # Get all files from samples directory
    samples_dir = Path("tests/samples")
    output_dir = Path("tests/output/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDF files
    sample_files = [
        f for f in samples_dir.iterdir() 
        if f.is_file() and f.suffix.lower() == '.pdf'
    ]

    if not sample_files:
        print(f"No PDF files found in {samples_dir}")
        return

    print(f"Found {len(sample_files)} PDF files to process")
    print("-" * 50)

    # Process each file
    for sample_file in sample_files:
        try:
            # Create a subdirectory for each file's images
            file_output_dir = output_dir / sample_file.stem

            # Convert document to images
            images = converter.convert_to_images(
                sample_file,
                output_dir=file_output_dir
            )

            print(f"✓ Successfully converted {sample_file.name}")
            print(f"  Number of pages: {len(images)}")
            print(f"  Output saved to {file_output_dir}")
            print(f"  First image size: {images[0].size}")
            print("-" * 50)

        except Exception as e:
            print(f"✗ Failed to convert {sample_file.name}")
            print(f"  Error: {str(e)}")
            print("-" * 50)


if __name__ == "__main__":
    test_document_to_images()
