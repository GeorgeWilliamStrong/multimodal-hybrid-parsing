import base64
import math
from pathlib import Path
from typing import List, Optional, Union
import io
from openai import OpenAI
from aiohttp import ClientSession
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    RetryError
)
from .heuristic_parser import DocumentParser
from tqdm.asyncio import tqdm


class HybridParser:
    """Parser class that combines heuristic parsing with VLM refinement"""

    def __init__(
        self,
        batch_size: int = 4,
        model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the hybrid parser

        Args:
            batch_size: Number of pages to process in each VLM batch
            model: OpenAI model to use
            openai_api_key: OpenAI API key. If not provided, will try to use
                environment variable.
        """
        self.batch_size = batch_size
        self.model = model
        self.heuristic_parser = DocumentParser()
        self.client = OpenAI(api_key=openai_api_key)

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(5)
    )
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
        system_msg = """
            Your task is to enhance and revise the provided Markdown text by accurately reflecting the content visible 
            in the document images. Make only the necessary corrections to ensure accuracy, formatting, and completeness, 
            without adding speculative content. Do not remove any text from the Drafted Markdown content.

            **Instructions:**
            1. **Review and Compare**: Carefully examine the details in both the document images and the Drafted Markdown 
            content. Ensure all text from the images is included in your revised Markdown.
            2. **Correct Formatting**: Fix any Markdown formatting errors while preserving all existing text, headings, 
            lists, math formulas (LaTeX), tables, charts, and inline formatting.
            3. **Detail Extraction**: Ensure all bullet points, numbered lists, Chinese numbers/lists, and tables are complete 
            and properly formatted. Pay special attention to the beginning of the document (i.e., texts in the beginning of 
            Drafted Markdown Content).
            4. **Image Descriptions**: Add accurate descriptions for all figures, images, and graphics contained in the document 
            images in detail using Markdown syntax, reflecting the content of these images. Where image tags (e.g., <!-- image -->) 
            and their corresponding images are present in the drafted markdown content, add detailed descriptions of the contents 
            of these images based on the corresponding figure in the document images.
            5. **Avoid Omissions**: Do not omit any text from the original document (i.e., drafted markdown content), especially 
            at the beginning of the pages. Ensure all content is extracted for completeness.
            6. **Output Requirements**: Provide the enhanced Markdown text without any code block markers or additional instructions.

            Ensure that all figures, images, and graphics contained in the document images are described in detail.
            Return only the enhanced/revised Markdown text. Do NOT output any code block symbols (```) and header/footer of each page.
        """

        # Prepare the prompt
        prompt_msg = f"""
            Conduct OCR to extract all text from the document images. Follow the guidelines below to enhance and correct the 
            Drafted Markdown content based on these images. Ensure strict alignment with the images and Drafted Markdown Content, 
            making only necessary corrections to improve accuracy and formatting.

            **Drafted Markdown Content**:
            ```
            {combined_text}
            ```

            **Guidelines:**
            1. **Text and Formatting**: Ensure all informative text and details are included. Correct any errors in the Markdown 
            text (e.g., line break (\\n), position, symbol, punctuation, etc.), maintaining proper structure and sentence with 
            appropriate headings and formatting.
            2. **Tables**: Format tables using pipes and dashes, ensuring completeness and alignment. Merge tables across images if necessary.
            3. **Images**: Add descriptions for each figure, images, or graphic, accurately reflecting their content. For diagrams or charts, provide 
            detailed descriptions of elements and data. Where image tags (e.g., <!-- image -->) and their corresponding images are present in the drafted 
            markdown content, add detailed descriptions of the contents of these images based on the corresponding figure in the document images.
            4. **Consistency**: Maintain consistent formatting and terminology throughout. Ensure the document is coherent, especially between heading levels.
            
            Ensure that all figures, images, and graphics contained in the document images are described in detail.
            Just output the enhanced Markdown text directly, without additional explanations or code block symbols (e.g., ```).
        """

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
                        "text": prompt_msg
                    },
                    *image_data
                ]
            }
        ]

        # Perform the API request asynchronously using aiohttp
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 8192,
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
        self,
        markdown_pages: List[str],
        image_bytes: List[bytes]
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

                tasks.append(
                    self._refine_batch_async(
                        batch_markdown,
                        batch_images,
                        session
                    )
                )

            # Await all tasks concurrently with progress bar
            refined_batches = await tqdm.gather(
                *tasks,
                desc="Processing batches",
                total=num_batches
            )
            return refined_batches

    async def parse_document(
        self,
        file_path: Union[str, Path]
    ) -> str:
        """
        Parse a document using the hybrid approach

        Args:
            file_path: Path to the document file

        Returns:
            str: Final refined markdown text
        """
        # Convert paths
        file_path = Path(file_path)

        # Process document and extract markdown and images
        self.heuristic_parser.process_document(file_path)
        markdown_pages = self.heuristic_parser.get_page_markdown()
        page_images = self.heuristic_parser.get_page_images()

        # Convert images to base64 encoded JPEGs
        image_bytes = []
        for img in page_images:
            img_byte_arr = io.BytesIO()
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            image_bytes.append(img_byte_arr)

        # Process batches concurrently using await
        try:
            refined_batches = await self._process_batch_concurrently(
                markdown_pages,
                image_bytes
            )
        except RetryError as e:
            # Handle failure after retries are exhausted
            print(f"Failed to process document after retries: {e}")
            return ""

        # Combine refined batches
        final_markdown = "\n\n".join(refined_batches)

        return final_markdown
