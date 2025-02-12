from pathlib import Path
from typing import Union, Optional
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)


# Configure the pipeline options
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 300 / 72.0
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True


class DocumentParser:
    """Parser class to convert documents to markdown using docling"""

    def __init__(self, device: Optional[str] = None, num_threads: int = 8):
        """
        Initialize the parser

        Args:
            device: Device for processing ('cuda', 'mps', 'cpu', or 'auto'). 
                If None, will use 'auto'.
            num_threads: Number of threads to use for processing
        """
        self.doc = None
        self.device = device or "auto"
        self.num_threads = num_threads

        # Map string device names to AcceleratorDevice enum
        device_map = {
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
            "cpu": AcceleratorDevice.CPU,
            "auto": AcceleratorDevice.AUTO,
        }

        if self.device not in device_map:
            raise ValueError(
                "Invalid device '{}'. Must be one of: {}".format(
                    device, list(device_map.keys())
                )
            )

        # Configure accelerator options
        self.accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=device_map[self.device]
        )

    def load_document(self, file_path: Union[str, Path]) -> None:
        """
        Load a document from a file path

        Args:
            file_path: Path to the document file
        """
        file_path = (
            Path(file_path) if isinstance(file_path, str) else file_path
        )

        # Update pipeline options with accelerator
        pipeline_options.accelerator_options = self.accelerator_options

        converter = DocumentConverter(
            format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
            }
        )
        self.doc = converter.convert(file_path)

    def to_markdown(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> str:
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
            output_path = (
                Path(output_path) 
                if isinstance(output_path, str) 
                else output_path
            )
            output_path.write_text("\n\n".join(markdown_pages))

        return markdown_pages
