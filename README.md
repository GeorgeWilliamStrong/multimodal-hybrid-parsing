# Multimodal Hybrid Parsing

A library for multimodal hybrid document parsing.

## System Requirements

Before installing this package, you need the following system dependencies:

- **Poppler** (for PDF processing)
  - macOS: `brew install poppler`
  - Ubuntu: `sudo apt-get install poppler-utils`
  - Windows: See [pdf2image documentation](https://github.com/Belval/pdf2image#windows)

## Installation

1. Install system dependencies (see above)

2. Install the Python package:
```bash
pip install multimodal-hybrid-parsing
```

## Usage

### Basic Document Parsing

For basic document parsing to markdown:

```python
from multimodal_hybrid_parsing import DocumentParser

# Create parser
parser = DocumentParser()

# Parse document
parser.load_document("path/to/document.pdf")
markdown = parser.to_markdown("output.md")  # output path is optional
```

### Image Extraction

To extract page images from a document:

```python
from multimodal_hybrid_parsing import PageImageConverter

# Create converter
converter = PageImageConverter(dpi=200)  # adjust DPI as needed

# Convert document to images
images = converter.convert_to_images(
    "path/to/document.pdf",
    "output/images"  # output directory is optional
)
```

### Hybrid Parsing

For enhanced document parsing using GPT-4V:

```python
from multimodal_hybrid_parsing import HybridParser

# Create hybrid parser (requires OpenAI API key)
parser = HybridParser(
    batch_size=3,  # number of pages per batch
    openai_api_key="your-api-key"  # optional, can use environment variable
)

# Parse document with GPT-4V refinement
markdown = parser.parse_document(
    "path/to/document.pdf",
    "output.md",  # output path is optional
    "temp/images"  # temporary image directory is optional
)
```

The hybrid parser combines traditional document parsing with GPT-4V's visual understanding to:
1. Fix text recognition errors
2. Improve formatting and structure
3. Add missing information from images
4. Maintain proper markdown formatting

## License

MIT
