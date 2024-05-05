import os
import io
import openai

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

        self.llm_for_cypher: ChatOpenAI = ChatOpenAI(
            model_name=self.deployment_name,
            temperature=0,
            model_kwargs={
                "engine": self.deployment_name
            }
        ) if llm_for_cypher is None else llm_for_cypher

        self.llm_for_rag: ChatOpenAI = ChatOpenAI(
            model_name=self.deployment_name,
            temperature=self.temperature,
            model_kwargs={
                "engine": self.deployment_name
            }
        ) if llm_for_rag is None else llm_for_rag
