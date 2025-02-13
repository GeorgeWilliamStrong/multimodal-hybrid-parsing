from pathlib import Path
from typing import Union, Optional
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    smolvlm_picture_description
)
from docling_core.types.doc import PictureItem


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

        # Configure pipeline options with picture description
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.images_scale = 300 / 72.0
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True

        # Enable picture description
        self.pipeline_options.do_picture_description = True
        self.pipeline_options.do_formula_enrichment = True
        self.pipeline_options.picture_description_options = smolvlm_picture_description
        self.pipeline_options.picture_description_options.generation_config = {
            "max_new_tokens": 1000,  # Maximum length of generated text
            "do_sample": False,      # Enable sampling
            "temperature": 0.7,     # Control randomness (higher = more random)
            "top_p": 0.9,          # Nucleus sampling parameter
            "top_k": 50,           # Top-k sampling parameter
            "repetition_penalty": 1.2  # Penalize repetition
        }

        # Optionally customize the prompt
        self.pipeline_options.picture_description_options.prompt = (
            "Describe the image in three sentences. Be concise and accurate."
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
        self.pipeline_options.accelerator_options = self.accelerator_options

        converter = DocumentConverter(
            format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipeline_options
                    )
            }
        )
        self.doc = converter.convert(file_path)
        for element, _level in self.doc.document.iterate_items():
            if isinstance(element, PictureItem):
                print(f"\nPicture {element.self_ref}")
                print(f"Caption: {element.caption_text(doc=self.doc.document)}")
                for annotation in element.annotations:
                    print(f"Description: {annotation.text}")

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

        # First collect all picture descriptions by page and position
        picture_descriptions = {}  # page_no -> list of descriptions in order of appearance
        for element, _level in self.doc.document.iterate_items():
            if isinstance(element, PictureItem):
                page_no = element.prov[0].page_no
                if page_no not in picture_descriptions:
                    picture_descriptions[page_no] = []
                
                # Get description or empty string if no annotations
                description = ""
                if element.annotations:
                    # Take only the first annotation for each image
                    ann = element.annotations[0]
                    description = f"**Image Description:** {ann.text}\n<!-- end image description -->"
                picture_descriptions[page_no].append(description)

        # Process each page
        markdown_pages = []
        for i in range(self.doc.document.num_pages()):
            page_no = i + 1
            page_md = self.doc.document.export_to_markdown(page_no=page_no)
            
            if page_no in picture_descriptions:
                # Split the markdown by image tags to process each section
                parts = page_md.split("<!-- image -->")
                
                # Reconstruct the page with descriptions
                new_page_md = parts[0]  # Start with the first part
                for idx, part in enumerate(parts[1:]):  # Process remaining parts
                    # Add image tag and description (if available)
                    if idx < len(picture_descriptions[page_no]):
                        description = picture_descriptions[page_no][idx]
                        if description:
                            new_page_md += f"<!-- image -->\n{description}\n{part}"
                        else:
                            new_page_md += f"<!-- image -->{part}"
                    else:
                        new_page_md += f"<!-- image -->{part}"
                
                page_md = new_page_md
            
            markdown_pages.append(page_md)

        # Write to file if output path provided
        if output_path:
            output_path = (
                Path(output_path) 
                if isinstance(output_path, str) 
                else output_path
            )
            output_path.write_text("\n\n".join(markdown_pages))

        return markdown_pages
