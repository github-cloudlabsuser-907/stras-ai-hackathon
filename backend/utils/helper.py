import os
import io
import openai
import logging
import re
import hashlib 
import requests, PyPDF4
import urllib
import time
from urllib import parse
from urllib.parse import unquote
from flask import Flask, Response, request, jsonify
from PyPDF4 import PdfFileReader
import nltk

nltk.download('punkt')
import pandas as pd

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter, NLTKTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain, OpenAI, PromptTemplate

from dotenv import load_dotenv
from .azureblobstorage import AzureBlobStorageClient
from .translator import AzureTranslatorClient
from .azuresearch import AzureSearch
from .formrecognizer import AzureFormRecognizerClient


import os
import chardet
import mimetypes
from http import HTTPStatus
from io import BytesIO
from urllib import parse
from docx import Document
from utils.documentparser import DocumentParser
from urllib.parse import unquote

from .customprompt import (
    PROMPT_KNOWLEDGE_SEARCH
)
class LLMHelper:
    def __init__(self,
                 llm_for_explanation: ChatOpenAI = None,
                 llm_for_project: ChatOpenAI = None,
                 llm_for_cypher: ChatOpenAI = None,
                 llm_for_rag: AzureOpenAI = None,
                 pdf_parser: AzureFormRecognizerClient = None,
                 
                 temperature: float = None,
                 max_tokens: int = None,
                 vector_store: VectorStore = None,
                 k: int = None,
                 blob_client: AzureBlobStorageClient = None,
                 enable_translation: bool = False,
                 translator: AzureTranslatorClient = None,
                 document_loaders: BaseLoader = None,
                 text_splitter: TextSplitter = None,
                 embeddings: OpenAIEmbeddings = None,
                 ):

        load_dotenv()
        openai.api_type = "azure"
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("WORKING_ENVIRONMENT") == "DEPLOYED":
            os.environ["NLTK_DATA"] = "/usr/local/share/nltk_data/tokenizers"

        self.chunk_size = int(os.getenv('CHUNK_SIZE', 2500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))
        # Azure OpenAI settings
        self.api_base = openai.api_base
        self.api_version = openai.api_version
        self.index_name: str = "effi-embeds-search"
        self.model_rag: str = "strasaiembeddings"
        self.deployment_name: str = os.getenv("OPENAI_ENGINE")
        self.temperature: float = float(os.getenv("OPENAI_TEMPERATURE", 0.7)) if temperature is None else temperature
        self.max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", -1)) if max_tokens is None else max_tokens

        self.pdf_parser: AzureFormRecognizerClient = AzureFormRecognizerClient() if pdf_parser is None else pdf_parser

        self.text_splitter: TextSplitter = NLTKTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        ) if text_splitter is None else text_splitter

        # Azure Search settings
        self.vector_store_address: str = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        self.vector_store_password: str = os.getenv('AZURE_SEARCH_ADMIN_KEY')

        self.chunk_size = int(os.getenv('CHUNK_SIZE', 2500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))

        self.document_loaders: BaseLoader = WebBaseLoader if document_loaders is None else document_loaders

        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            deployment=self.model_rag,
            chunk_size=1,
            openai_api_type="azure",
        ) if embeddings is None else embeddings

        self.vector_store: VectorStore = AzureSearch(azure_cognitive_search_name=self.vector_store_address,
                                                     azure_cognitive_search_key=self.vector_store_password,
                                                     index_name=self.index_name,
                                                     embedding_function=self.embeddings.embed_query
                                                     ) if vector_store is None else vector_store
        self.k: int = 3 if k is None else k

        self.blob_client: AzureBlobStorageClient = AzureBlobStorageClient() if blob_client is None else blob_client
        self.enable_translation: bool = False if enable_translation is None else enable_translation
        self.translator: AzureTranslatorClient = AzureTranslatorClient() if translator is None else translator

        # LLM initialization upon the needs
        self.llm_for_explanation: ChatOpenAI = ChatOpenAI(
            model_name=self.deployment_name,
            temperature=self.temperature,
            model_kwargs={
                "engine": self.deployment_name
            }
        ) if llm_for_explanation is None else llm_for_explanation

        self.llm_for_project: ChatOpenAI = ChatOpenAI(
            model_name=self.deployment_name,
            temperature=0.5,
            model_kwargs={
                "engine": self.deployment_name
            }
        ) if llm_for_project is None else llm_for_project

    def knowledge_search(
        self,
        u_chapter: str,
        u_question: str
    ) -> str:
        print("knowledge search")
        result_search = LLMChain(
            prompt=PROMPT_KNOWLEDGE_SEARCH,
            llm=self.llm_for_project,
            verbose=True
        ).run(
            context=u_chapter,
            question=u_question
        )
        return result_search

    
    def upload_document(self) -> tuple[Response, HTTPStatus]:
        """
        Upload et embedded un fichier
        :return: Retourne une response
        :rtype: Response
        """
        file = os.path.basename("input_doc.docx")
        try:
    
            #uploaded_file_bytes = file
            with open("input_doc.docx", "rb") as f:
                
                b = BytesIO(f.read())
                

                # Creation of file URL
            content_type = mimetypes.guess_type(file)[0]
            charset = f"; charset={chardet.detect(b)['encoding']}" if content_type == 'text/plain' else ''

            converted_chapters_name = []

            document_parsed = DocumentParser(
                file_name=file.split('.')[0],
                pdf_doc_stream=b
            ).create_stream_from_doc()
            for chapter_name, chapter_content in document_parsed.items():
                try:   
                    # Upload the file on the Azure Blob Storage
                    if file is not None:
                        source_url = self.blob_client.upload_file(
                            bytes_data=chapter_content,
                            file_name=chapter_name,
                            content_type=content_type + charset
                        )
                        converted_filename = ''
                        if file.endswith('.txt'):
                            # Add the text to the embeddings
                            self.add_embeddings_lc(source_url)

                        else:
                            # Get OCR with Layout API and then add embeddigns
                            converted_chapters_name = self.convert_file_and_add_embeddings(source_url, chapter_name,
                                                                                                False)
                        self.blob_client.upsert_blob_metadata(chapter_name,
                                                                    {'converted': 'true', 'embeddings_added': 'true',
                                                                    'converted_filename': parse.quote(
                                                                        converted_chapters_name)})
                except Exception as error:
                    print("An exception occurred:", error) 
                    return jsonify(status="ok", description="File uploaded and embedded successfully !"), HTTPStatus.MULTI_STATUS
        except Exception as e:
            logging.error(f"Error occurred during upload the document: {str(e)}")
            return jsonify(
            status="ko",
            description=f"Error occurred during upload the document: {str(e)}"
            ), HTTPStatus.MULTI_STATUS
        
    def convert_file_and_add_embeddings(self, source_url, filename, enable_translation=False):

        # Extract the text from the file
        converted_text = self.pdf_parser.analyze_read(source_url)

        # Remove half non-ascii character from start/end of doc content
        # (langchain TokenTextSplitter may split a non-ascii character in half)
        pattern = re.compile(
            # do not remove \x0a (\n) nor \x0d (\r)
            r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f\u0080-\u00a0\u2000-\u3000\ufff0-\uffff]'
        )
        converted_text = re.sub(pattern, '', "\n".join(converted_text))
        # Upload the text to Azure Blob Storage
        converted_filename = f"converted/{filename}.txt"
        source_url = self.blob_client.upload_file(converted_text, f"converted/{filename}.txt",
                                                  content_type='text/plain; charset=utf-8')
        # Update the metadata to indicate that the file has been converted
        self.blob_client.upsert_blob_metadata(filename, {"converted": "true"})

        self.add_embeddings_lc(source_url=source_url)

        return converted_filename
    
    def add_embeddings_lc(self, source_url):
        try:
            documents = self.document_loaders(source_url).load()
            # Convert to UTF-8 encoding for non-ascii text
            for (document) in documents:
                try:
                    if document.page_content.encode("iso-8859-1") == document.page_content.encode("latin-1"):
                        document.page_content = document.page_content.encode("iso-8859-1").decode("utf-8",
                                                                                                  errors="ignore")
                except:
                    pass

            docs = self.text_splitter.split_documents(documents)

            # Remove half non-ascii character from start/end of doc content (langchain TokenTextSplitter may split a non-ascii character in half)
            pattern = re.compile(
                r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f\u0080-\u00a0\u2000-\u3000\ufff0-\uffff]')  # do not remove \x0a (\n) nor \x0d (\r)
            for (doc) in docs:
                doc.page_content = re.sub(pattern, '', doc.page_content)
                if doc.page_content == '':
                    docs.remove(doc)
            keys = []
            for i, doc in enumerate(docs):
                # Create a unique key for the document
                source_url = source_url.split('?')[0]
                filename = "/".join(source_url.split('/')[4:])
                hash_key = hashlib.sha1(f"{source_url}_{i}".encode('utf-8')).hexdigest()
                hash_key = f"doc:{self.index_name}:{hash_key}"
                keys.append(hash_key)
                doc.metadata = {"source": f"[{source_url}]({source_url}_SAS_TOKEN_PLACEHOLDER_)", "chunk": i,
                                "key": hash_key, "filename": filename}

            self.vector_store.add_documents(documents=docs, keys=keys)
        except Exception as e:
            logging.error(f"Error adding embeddings for {source_url}: {e}")
            raise e
        

    def similarity_search_with_score(self, prompt_result: str, k: int = None, threshold : float = None):

        result = self.vector_store.similarity_search_with_score(query=prompt_result, k=k if k else self.k)
        threshold = threshold if threshold else 0.1

        dataFrame = pd.DataFrame([
            {
                'key': x[0].metadata['key'],
                'filename': x[0].metadata['filename'],
                'source': urllib.parse.unquote(x[0].metadata['source']),
                'content': x[0].page_content,
                'metadata': x[0].metadata,
                'score': x[1] if len(x) > 1 else None
            }
            for x in result if (len(x) > 1 and x[1] > threshold)
        ])

        if not dataFrame.empty:
            dataFrame = dataFrame.sort_values(by='score')

        return dataFrame

    def group_content(self, evol_similarity_result)->list:
            """
            Group similar contents from the evolution similarity result.
    
            Parameters:
            - evol_similarity_result: DataFrame
                A DataFrame containing the similarity results with 'filename' and 'content' columns.
    
            Returns:
            dict:
                A list of dictionaries containing filenames, scores, and aggregated contents as values.
            """
            # Convert DataFrame to a list of dictionaries
            records = evol_similarity_result.to_dict(orient='records')
    
            # Initialize list of dictionaries to hold the unique filenames, scores and contents
            unique_content_dict = []

            # Iterate through the records
            for record in records:
                content_dict = {}
                filename = record['filename']
                content = record['content']
                score = record['score']

                content_dict = {'filename': filename, 'score': score, 'content': content}
                unique_content_dict.append(content_dict)

            return unique_content_dict


    def llm_knowledge_search(self, context: str, question: str) -> str:
        result = LLMChain(
            prompt=PROMPT_KNOWLEDGE_SEARCH,
            llm=self.llm_for_project,
            verbose=True
        ).run(
            query=question,
            context=context
            )
        return result