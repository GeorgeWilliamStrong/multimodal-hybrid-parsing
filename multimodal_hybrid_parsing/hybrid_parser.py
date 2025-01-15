import base64
import math
from pathlib import Path
from typing import List, Optional, Union
import io
from openai import OpenAI
from aiohttp import ClientSession
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from .heuristic_parser import DocumentParser
from .page_images import PageImageConverter
from tqdm.asyncio import tqdm


class HybridParser:
    """Parser class that combines heuristic parsing with GPT-4V refinement"""

    def __init__(
        self, batch_size: int = 3, openai_api_key: Optional[str] = None
    ):
        """
        Initialize the hybrid parser

        Args:
            batch_size: Number of pages to process in each GPT-4V batch
            openai_api_key: OpenAI API key. If not provided, will try to use
                environment variable.
        """
        self.batch_size = batch_size
        self.heuristic_parser = DocumentParser()
        self.image_converter = PageImageConverter()
        self.client = OpenAI(api_key=openai_api_key)

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _refine_batch_async(
        self,
        markdown_texts: List[str],
        images: List[bytes],
        session: ClientSession
    ) -> str:
        """
        Refine a batch of markdown texts using GPT-4o-mini asynchronously.

        Args:
            markdown_texts: List of markdown texts to refine
            images: List of corresponding page images
            session: aiohttp session for async requests

        Returns:
            str: Refined markdown text
        """
        # Combine markdown texts
        combined_text = "\n\n".join(markdown_texts)

        # Create messages for GPT-4o-mini
        system_msg = (
            "You are an expert document parser. Your task is to refine and "
            "improve the provided markdown text using the visual information "
            "from the page images. Focus on:\n"
            "1. Fixing any text recognition errors\n"
            "2. Improving formatting and structure\n"
            "3. Adding missing information visible in the images\n"
            "4. Maintaining proper markdown formatting\n"
            "Return only the refined markdown text without any explanations."
        )

        # Prepare the image data in base64
        image_data = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(image).decode('utf-8')}",
                    "detail": "high"
                }
            }
            for image in images
        ]

        # Prepare the message with markdown text and images
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is the current markdown text:\n\n{combined_text}"
                    },
                    *image_data
                ]
            }
        ]

        # Perform the API request asynchronously using aiohttp
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0
            },
            headers={"Authorization": f"Bearer {self.client.api_key}"}
        ) as response:
            data = await response.json()

            # Check if response contains errors or rate limit exceeded
            if 'error' in data:
                raise Exception(f"API Error: {data['error']['message']}")

            return data['choices'][0]['message']['content']

    async def _process_batch_concurrently(
        self, markdown_pages: List[str], image_bytes: List[bytes]
    ) -> List[str]:
        """
        Process batches concurrently.

        Args:
            markdown_pages: List of markdown texts
            image_bytes: List of base64-encoded images

        Returns:
            List[str]: List of refined markdown texts
        """
        async with ClientSession() as session:
            tasks = []
            num_pages = len(markdown_pages)
            num_batches = math.ceil(num_pages / self.batch_size)

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_pages)

                batch_markdown = markdown_pages[start_idx:end_idx]
                batch_images = image_bytes[start_idx:end_idx]

                tasks.append(self._refine_batch_async(batch_markdown, batch_images, session))

            # Await all tasks concurrently with progress bar
            refined_batches = await tqdm.gather(
                *tasks,
                desc="Processing batches",
                total=num_batches
            )
            return refined_batches

    async def parse_document(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        temp_image_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Parse a document using the hybrid approach

        Args:
            file_path: Path to the document file
            output_path: Optional path to save the final markdown output
            temp_image_dir: Optional directory to save temporary page images

        Returns:
            str: Final refined markdown text
        """
        # Convert paths
        file_path = Path(file_path)
        if output_path:
            output_path = Path(output_path)
        if temp_image_dir:
            temp_image_dir = Path(temp_image_dir)

        # Extract markdown and images
        self.heuristic_parser.load_document(file_path)
        markdown_pages = self.heuristic_parser.to_markdown()
        images = self.image_converter.convert_to_images(
            file_path, temp_image_dir
        )

        # Convert images to base64 encoded JPEGs
        image_bytes = []
        for img in images:
            img_byte_arr = io.BytesIO()
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            image_bytes.append(img_byte_arr)

        # Process batches concurrently using await
        try:
            refined_batches = await self._process_batch_concurrently(markdown_pages, image_bytes)
        except RetryError as e:
            # Handle failure after retries are exhausted
            print(f"Failed to process document after retries: {e}")
            return ""

        # Combine refined batches
        final_markdown = "\n\n".join(refined_batches)

        # Save output if path provided
        if output_path:
            output_path.write_text(final_markdown)

        return final_markdown
