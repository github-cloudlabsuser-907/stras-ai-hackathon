
from typing import Any
import io
from .azureblobstorage import AzureBlobStorageClient
from docx import Document
from docx.oxml import CT_Picture
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from urllib.parse import unquote


class DocumentParser:
    """
    A class for parsing documents, extracting sections, and creating PDF streams from the document content.

    Attributes:
        file_name (str): The name of the document file.
        pdf_doc_stream: The PDF document stream.
        blob_client (AzureBlobStorageClient): An instance of AzureBlobStorageClient.
    """

    def __init__(self, file_name: str, pdf_doc_stream, blob_client: AzureBlobStorageClient = None):
        """
        Initialize the DocumentParser class.

        Args:
            file_name (str): The name of the document file.
            pdf_doc_stream: The PDF document stream.
            blob_client (AzureBlobStorageClient, optional): An instance of AzureBlobStorageClient. Defaults to None.
        """
        self.file_name = file_name 
        self.pdf_doc_stream = pdf_doc_stream 
        #self.blob_client: AzureBlobStorageClient = AzureBlobStorageClient() if blob_client is None else blob_client
    
    
    def iter_block_items(self, parent):
        """
        Yield each paragraph and table child within *parent*, in document order.
        """
        from docx.document import Document
        if isinstance(parent, Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("Something's not right")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def get_level(self, paragraph)-> str|None:
        """
        Function to get the level of a paragraph.
        """
        style = paragraph.style.name
        if style.startswith("Heading 4"):
            return "header_4"
        elif style.startswith("Heading 3"):
            return "header_3"
        elif style.startswith("Heading 2"):
            return "header_2"
        elif style.startswith("Heading 1"):
            return "header_1"
        else:
            return None
        
    def parse_document(self):

        # List to store the hierarchical structure of the document
        chapter_structure = []
        doc = Document(self.pdf_doc_stream)
        # Iterate through paragraphs and store the hierarchical structure
        current_sections = [0, 0, 0, 0]  # To track the current section number at each level
        for item in self.iter_block_items(doc):
            if isinstance(item, Paragraph):
                level = self.get_level(item)
                if level:
                    level_index = int(level[-1]) - 1  # Convert 'header_#' to an index
                    for i in range(level_index + 1, 4):
                        current_sections[i] = 0  # Reset lower level sections
                    current_sections[level_index] += 1  # Increment the corresponding section number
                    section_title = '.'.join(str(num) for num in current_sections[:level_index + 1]) + " " + item.text
                    chapter_structure.append({"title": section_title, "content": [], "tables": []})
                elif chapter_structure:  # Add paragraph content to the latest section
                    chapter_structure[-1]["content"].append(item.text)

            elif isinstance(item, Table) and chapter_structure:
                    chapter_structure[-1]["tables"].append(item)

        return chapter_structure
