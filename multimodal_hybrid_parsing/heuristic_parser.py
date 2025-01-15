from docling.document_converter import DocumentConverter
from pathlib import Path
from typing import Union, Optional


class DocumentParser:
    """Parser class to convert documents to markdown using docling"""

    def __init__(self):
        self.doc = None

    def load_document(self, file_path: Union[str, Path]) -> None:
        """
        Load a document from a file path

        Args:
            file_path: Path to the document file
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        converter = DocumentConverter()
        self.doc = converter.convert(file_path)

    def to_markdown(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert the loaded document to markdown

        Args:
            output_path: Optional path to save the markdown output

        Returns:
            str: Markdown representation of the document
        """
        if not self.doc:
            raise ValueError("No document loaded. Call load_document() first.")

        markdown_pages = [
            self.doc.document.export_to_markdown(page_no=i + 1)
            for i in range(self.doc.document.num_pages())
        ]

        if output_path:
            output_path = Path(output_path) if isinstance(output_path, str) else output_path
            output_path.write_text("\n\n".join(markdown_pages))

        return markdown_pages
