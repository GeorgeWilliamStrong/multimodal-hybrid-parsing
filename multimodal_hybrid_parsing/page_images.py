from pathlib import Path
from typing import List, Optional, Union
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image


class PageImageConverter:
    """Converts PDF documents to page images."""
    
    def __init__(self, dpi: int = 300):
        """Initialize the converter with specified DPI for image conversion.
        
        Args:
            dpi: Dots per inch for the output images. Higher values mean better
                quality but larger files.
        """
        self.dpi = dpi
        
    def convert_to_images(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Image.Image]:
        """Convert a PDF file to a list of PIL Image objects, one per page.
        
        Args:
            file_path: Path to the PDF file
            output_dir: Optional directory to save images to. If provided, saves
                each page as '{page_num}.png' in this directory.
            
        Returns:
            List of PIL Image objects, one for each page
            
        Raises:
            ValueError: If the file is not a PDF or doesn't exist
            RuntimeError: If conversion fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() != '.pdf':
            raise ValueError("Only PDF files are supported")

        # Convert PDF to images
        try:
            images = convert_from_path(
                file_path,
                dpi=self.dpi,
                fmt='PIL'
            )
        except PDFPageCountError as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
                
        # Save images if output directory is specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images, start=1):
                image.save(output_dir / f"{i}.png")
                
        return images
