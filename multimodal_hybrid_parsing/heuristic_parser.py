from pathlib import Path
from typing import Union, Optional, List, Tuple
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions
)
from docling_core.types.doc import PictureItem
from docling.pipeline.simple_pipeline import SimplePipeline
from PIL import Image


class DocumentParser:
    """Parser class to convert documents to markdown using docling"""

    def __init__(
        self,
        device: Optional[str] = None,
        num_threads: int = 8,
        images_scale: float = 300/72.0,
    ):
        """
        Initialize the parser

        Args:
            device: Device for processing ('cuda', 'mps', 'cpu', or 'auto'). 
                If None, will use 'auto'.
            num_threads: Number of threads to use for processing
            images_scale: Scale factor for images (default: 300/72.0)
        """
        self.doc = None
        self.device = device or "auto"
        self.num_threads = num_threads
        self.images_scale = images_scale

        # Map string device names to AcceleratorDevice enum
        device_map = {
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS, 
            "cpu": AcceleratorDevice.CPU,
            "auto": AcceleratorDevice.AUTO,
        }

        if self.device not in device_map:
            msg = f"Invalid device '{device}'. Must be one of: {list(device_map.keys())}"
            raise ValueError(msg)

        # Configure pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.images_scale = self.images_scale
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_formula_enrichment = True

        # Configure accelerator options
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=device_map[self.device]
        )

        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.PPTX,
                InputFormat.DOCX
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline
                )
            }
        )

    def process_document(self, file_path: Union[str, Path]) -> None:
        """
        Process a document

        Args:
            file_path: Path to the document file
        """
        file_path = (
            Path(file_path) if isinstance(file_path, str) else file_path
        )

        self.doc = self.converter.convert(file_path)

    def get_page_images(self) -> List[Image.Image]:
        """
        Get a list of page images from the processed document

        Returns:
            List of PIL Image objects, one for each page
        """
        page_images = []
        for _, page in self.doc.document.pages.items():
            if hasattr(page, 'image') and page.image is not None:
                page_images.append(page.image.pil_image)
        return page_images

    def extract_images(self) -> Tuple[List[Image.Image], List[int]]:
        """
        Extract images from the processed document

        Returns:
            Tuple containing:
            - List of extracted images
            - List of page numbers where images were found
        """
        extracted_images = []
        pages_with_images = []
        # Extract images
        for element, _ in self.doc.document.iterate_items():
            if isinstance(element, PictureItem):
                # For DOCX files, we'll use a default page number of 1 since the number of pages aren't properly registered (Docling bug)
                page_no = element.prov[0].page_no if element.prov else 1
                if hasattr(element, 'image') and element.image is not None:
                    extracted_images.append(element.image.pil_image)
                    pages_with_images.append(page_no)
        return extracted_images, pages_with_images

    def get_page_markdown(self) -> List[str]:
        """
        Convert the processed document to markdown

        Returns:
            List[str]: List of markdown pages
        """
        if not self.doc:
            raise ValueError("No document processed. Call process_document() first.")

        markdown_pages = []
        # For DOCX files, we need to handle the case where num_pages() returns 0
        if self.doc.document.num_pages() == 0:
            # Get markdown for the entire document as a single page
            full_md = self.doc.document.export_to_markdown()
            markdown_pages.append(full_md)
        else:
            # Process markdown pages normally for PDF and PPTX
            for page in range(self.doc.document.num_pages()):
                page_no = page + 1
                page_md = self.doc.document.export_to_markdown(page_no=page_no)
                markdown_pages.append(page_md)

        return markdown_pages
