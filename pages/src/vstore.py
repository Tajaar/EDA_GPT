import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
import tempfile
from io import BytesIO
import logging
from functools import wraps

import nltk
import pandas as pd
import PyPDF2
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    JSONLoader,
    MergedDataLoader,
)
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceHubEmbeddings,
    GPT4AllEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from unstructured.partition.pdf import partition_pdf
import chromadb
from chromadb.config import Settings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
# Initialize NLTK resources with retries
def init_nltk_resources(max_retries=3):
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        for attempt in range(max_retries):
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Downloaded NLTK resource: {resource}")
                break
            except Exception as e:
                logger.warning(f"Failed to download {resource} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to download NLTK resource {resource} after {max_retries} attempts")

init_nltk_resources()

# Streamlit compatibility wrapper
def streamlit_safe(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NameError as e:
            if "st" in str(e):
                logger.warning(f"Streamlit not available, skipping {func.__name__}")
                return lambda x: None  # Return no-op for spinners/warnings
            raise
    return wrapper

class VectorStore:
    def __init__(self, directory=None, use_streamlit=True, **kwargs):
        self.config_file_path = os.path.join("pages", "src", "Database", "config.json")
        self.use_streamlit = use_streamlit
        try:
            with open(self.config_file_path, "r") as file:
                self.config_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load config file: {e}")
            raise

        self.unstructured_directory_path = self.config_data.get("unstructured_data", "")
        self.vector_stores = None
        self.directory = directory
        self.data = None
        self.chroma_db = None
        self.vector_stores_list = []
        self.embedding_type = None  # Track embedding type for serialization

        if self.directory:
            self.unstructured_directory_path = self.directory

    def get_embedding_function(self, embedding_num=1):
        """Helper to initialize embedding functions with robust secret handling."""

        def get_secret(key, env_var):
            secret = st.secrets.get(key, "") if self.use_streamlit else ""
            env_secret = secret or os.getenv(env_var)
            if not env_secret:
                raise ValueError(f"Missing {env_var} in environment or Streamlit secrets")
            return env_secret

        try:
            embedding_options = {
                0: HuggingFaceInferenceAPIEmbeddings(
                    api_key=get_secret("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
                    model_name="BAAI/bge-base-en-v1.5",
                ),
                1: GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=get_secret("GOOGLE_GEMINI_API", "GOOGLE_API_KEY"),
                ),
                2: HuggingFaceHubEmbeddings(
                    huggingfacehub_api_token=get_secret("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
                ),
                3: OpenAIEmbeddings(
                    api_key=get_secret("OPENAI_API_KEY", "OPENAI_API_KEY"),
                ),
                4: GPT4AllEmbeddings(),
            }

            embeddings = embedding_options.get(embedding_num)
            if embeddings is None:
                logger.warning(f"Invalid embedding_num={embedding_num}, defaulting to GoogleGenerativeAI")
                embeddings = embedding_options[1]

            self.embedding_type = embedding_num
            logger.info(f"Using embeddings: {embeddings.__class__.__name__}")
            return embeddings

        except Exception as e:
            logger.error(f"❌ Error initializing embedding function: {e}")
            raise


    @streamlit_safe
    def makevectorembeddings(
        self, key: Optional[str] = None, embedding_num: int = 1, 
        use_semantic_chunker: bool = False, **kwargs
    ) -> Tuple[Chroma, BM25Retriever]:
        try:
            import shutil
            import uuid
            import time
            import subprocess
            import platform
            from langchain_chroma import Chroma
            from chromadb.config import Settings

            persist_directory = "./chroma_db"

            def safe_delete(path):
                if os.path.exists(path):
                    try:
                        if platform.system() == "Windows":
                            subprocess.call(['cmd', '/c', 'rmdir', '/S', '/Q', path])
                        else:
                            shutil.rmtree(path)
                        logger.info(f"🧹 Cleared existing Chroma directory: {path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error deleting directory {path}: {e}")

            safe_delete(persist_directory)

            if not self.directory and key is None:
                with st.spinner("🔎 Searching for structure in the data..."):
                    self._preprocess_data_in_directory()

            with st.spinner("📅 Loading data as text..."):
                text_loader = DirectoryLoader(
                    self.unstructured_directory_path, glob="*.txt", show_progress=True
                )
                if key is None:
                    csv_loader = DirectoryLoader(
                        self.unstructured_directory_path,
                        glob="**/[!.]*.csv",
                        show_progress=True,
                    )
                    merged_data_loader = MergedDataLoader([text_loader, csv_loader])
                    self.data = merged_data_loader.load()
                else:
                    self.data = text_loader.load()

            self.data = [doc for doc in self.data if doc.page_content.strip()]
            if not self.data:
                raise ValueError("❌ No valid documents loaded. Cannot proceed with embeddings.")

            embeddings = self.get_embedding_function(embedding_num)
            collection_name = f"collection_{int(time.time())}_{uuid.uuid4().hex}"

            settings = Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )

            if self.directory is None:
                with st.spinner("🔧 Creating chunks for unstructured data..."):
                    chunks = self.create_chunks_for_parallel_processing(self.data, use_semantic_chunker)
                    list_of_docs = self.documents_from_chunks(text_chunks=chunks)
                    list_of_docs = [doc for doc in list_of_docs if doc and doc.page_content.strip()]

                    if not list_of_docs:
                        logger.error("❌ No valid documents to embed. Aborting Chroma creation.")
                        raise ValueError("No valid documents found to embed.")

                    logger.debug(f"📊 Total documents to embed: {len(list_of_docs)}")

                with st.spinner("📌 Creating embeddings and adding to Chroma..."):
                    try:
                        logger.info("🚀 Creating Chroma vector store...")
                        logger.debug(f"📁 Collection: {collection_name}, Docs: {len(list_of_docs)}")
                        self.vector_store = Chroma.from_documents(
                            documents=list_of_docs,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                            collection_name=collection_name,
                            client_settings=settings
                        )
                        logger.info("✅ Chroma vector store created successfully.")
                    except Exception as e:
                        logger.error(f"🔥 Error during Chroma creation: {str(e)}", exc_info=True)
                        raise

                with st.spinner("📚 Creating BM25 retriever..."):
                    bm25_retriever = BM25Retriever.from_documents(list_of_docs)

                return self.vector_store, bm25_retriever

            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=100, length_function=len
                ) if not use_semantic_chunker else SemanticChunker(embeddings)

                chunks = text_splitter.split_documents(documents=self.data)

                for i, doc in enumerate(self.data[:5]):
                    logger.debug(f"📄 Original doc {i}: {doc.page_content[:200]}")

                if not chunks:
                    logger.error("❌ No document chunks created for structured data.")
                    raise ValueError("No document chunks were created.")

                try:
                    logger.info("🚀 Creating ChromaDB for structured data")
                    logger.debug(f"📁 Collection: {collection_name}, Chunks: {len(chunks)}")
                    self.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=persist_directory,
                        collection_name=collection_name,
                        client_settings=settings
                    )
                    logger.info("✅ Chroma vector store created successfully.")
                except Exception as e:
                    logger.error(f"🔥 Error during Chroma creation: {str(e)}", exc_info=True)
                    raise

                bm25_retriever = BM25Retriever.from_documents(chunks)
                return self.vector_store, bm25_retriever

        except ValueError as ve:
            logger.error(f"🚨 ValueError in makevectorembeddings: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"🔥 General error in makevectorembeddings: {str(e)}", exc_info=True)
            raise


    def _create_documents_in_parallel(self, text_chunk):
        """Creates Langchain documents from text chunks in parallel."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20, length_function=len
        )
        return splitter.create_documents([text_chunk])

    def documents_from_chunks(self, text_chunks):
        """
        Creates Langchain documents from a list of text chunks.

        Args:
            text_chunks (list): A list of text chunks.

        Returns:
            list: A list of Langchain documents.
        """
        if not text_chunks:
            print("⚠️ Warning: text_chunks is empty!")
            return []

        max_workers = os.cpu_count() or 1

        chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._create_documents_in_parallel, chunk)
                for chunk in text_chunks
            ]
            for future in futures:
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    print(f"⚠️ Thread failed: {e}")
        return chunks

    def create_chunks_for_parallel_processing(self, data, use_semantic_chunker=False):
        text = "\n".join(doc.page_content for doc in data if doc.page_content.strip())
        if not text.strip():
            logger.warning("No valid text found")
            return []

        logger.info(f"Total text length: {len(text)}")
        total_length = len(text)
        chunk_size = max(500, {  # Minimum chunk size
            100_000_000: 1_000_000,
            10_000_000: 100_000,
            1_000_000: 10_000,
            10_000: 2000,
            100: 500,
        }.get(next((x for x in sorted([100_000_000, 10_000_000, 1_000_000, 10_000, 100]) if x <= total_length), 100)))
        chunk_overlap = min(20, chunk_size // 10)

        text_splitter = (SemanticChunker(self.get_embedding_function(self.embedding_type))
                         if use_semantic_chunker else
                         RecursiveCharacterTextSplitter(
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
                         ))
        splitted_text_list = text_splitter.split_text(text)
        if not splitted_text_list:
            logger.warning("No text was split into chunks")
            return []

        num_chunks = os.cpu_count() or 1
        chunked_lists = [splitted_text_list[i::num_chunks] for i in range(num_chunks)]
        text_chunks = [" ".join(group) for group in chunked_lists if group]
        return text_chunks

    def _create_batches(self, list_of_docs):
        """Creates batches of documents for parallel processing."""
        cpu_count = os.cpu_count() or 1
        batch_size = max(len(list_of_docs) // cpu_count, 1)
        batches = [
            list_of_docs[i : i + batch_size]
            for i in range(0, len(list_of_docs), batch_size)
        ]
        return batches


    def _add_to_chroma(self, collection, chunks, batch_index):
        try:
            collection.add(
                documents=[doc.page_content for doc in chunks],
                metadatas=[
                    doc.metadata if doc.metadata else {"source": "unknown"}
                    for doc in chunks
                ],
                ids=[f"{batch_index}_{i}" for i in range(len(chunks))],
            )
            return True
        except Exception as e:
            logger.error(f"Error in _add_to_chroma: {e}")
            return False


    def _parallel_embeddings(self, db, list_of_docs, embeddings, collection_name):
        batches = self._create_batches(list_of_docs)
        if not batches:
            logger.warning("No batches created for parallel embedding")
            return None

        # Wrap LangChain embeddings for Chroma
        def chroma_embedding_function(texts):
            return embeddings.embed_documents(texts)

        try:
            collection = db.get_or_create_collection(
                name=collection_name, embedding_function=chroma_embedding_function
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chroma collection: {e}")
            return None

        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures = [
                executor.submit(self._add_to_chroma, collection, batch, idx)
                for idx, batch in enumerate(batches)
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Thread failed: {e}")

        logger.info(f"Parallel embedding completed. Successful batches: {sum(results)}/{len(batches)}")
        return collection

    def _savevectorstores(self):
        if not self.vector_stores:
            logger.error("Cannot serialize `self.vector_stores` because it's None")
            return

        serialized_path = os.path.join(self.unstructured_directory_path, "serialized_index.pkl")
        if isinstance(self.vector_stores, FAISS):
            try:
                serialized_data = self.vector_stores.serialize_to_bytes()
                with open(serialized_path, "wb") as file:
                    pickle.dump({"data": serialized_data, "embedding_type": self.embedding_type}, file)
                logger.info(f"Successfully saved FAISS index to {serialized_path}")
            except Exception as e:
                logger.error(f"Error serializing FAISS index: {e}")
        elif isinstance(self.vector_stores, Chroma):
            logger.info("Chroma vector store saved implicitly via PersistentClient")
        else:
            logger.warning("Serialization not supported for this vector store type")


    def loadvectorstores(self, embedding_num=1):
        serialized_path = os.path.join(self.unstructured_directory_path, "serialized_index.pkl")
        if not os.path.exists(serialized_path):
            logger.error(f"Serialized FAISS index not found at {serialized_path}")
            return None

        try:
            with open(serialized_path, "rb") as file:
                serialized_data = pickle.load(file)
                serialized_faiss = serialized_data["data"]
                stored_embedding_type = serialized_data.get("embedding_type", embedding_num)
            logger.info("Successfully loaded serialized FAISS index")
        except Exception as e:
            logger.error(f"Error loading serialized FAISS file: {e}")
            return None

        embeddings = self.get_embedding_function(stored_embedding_type)
        try:
            self.vector_stores = FAISS.deserialize_from_bytes(
                serialized_faiss, embeddings=embeddings
            )
            self.embedding_type = stored_embedding_type
            logger.info("FAISS deserialization successful")
        except Exception as e:
            logger.error(f"Error deserializing FAISS: {e}")
            self.vector_stores = None
        return self.vector_stores


    def _preprocess_text(self, text):
        """
        Preprocesses the input text by converting to lowercase, removing non-word
        characters, removing punctuation and digits, removing stop words, and
        lemmatizing.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The cleaned and preprocessed text.
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and digits
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)

        # Tokenize
        words = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in stop_words and word.isalpha()]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        # Join back into a single string
        return " ".join(lemmatized_words)


    def split_pdf_in_memory(self, input_pdf_path):
        """
        Splits a PDF file into multiple parts, stored as temporary files in memory.

        Args:
            input_pdf_path (str): The path to the input PDF file.

        Returns:
            list: A list of paths to the temporary PDF parts.
        """
        if not os.path.exists(input_pdf_path):
            print(f"⚠️ Error: PDF file not found at {input_pdf_path}")
            return []

        try:
            pdf_reader = PyPDF2.PdfReader(input_pdf_path)
        except Exception as e:
            print(f"⚠️ Error reading PDF file: {e}")
            return []

        num_pages = len(pdf_reader.pages)
        if num_pages == 0:
            print("⚠️ Error: PDF has no pages.")
            return []

        num_cores = min(os.cpu_count() or 1, num_pages)  # Prevent ZeroDivisionError
        pages_per_part = max(num_pages // num_cores, 1)  # Ensure at least 1 page per part

        pdf_parts = []
        current_part_pages = 0
        pdf_writer = None

        for i in range(num_pages):
            if current_part_pages == 0:
                pdf_writer = PyPDF2.PdfWriter()

            pdf_writer.add_page(pdf_reader.pages[i])
            current_part_pages += 1

            if current_part_pages == pages_per_part or i == num_pages - 1:
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as temp_pdf:
                        pdf_writer.write(temp_pdf)
                        temp_pdf_path = temp_pdf.name
                    pdf_parts.append(temp_pdf_path)
                except Exception as e:
                    print(f"⚠️ Error creating temporary PDF file: {e}")
                current_part_pages = 0  # Reset for next batch
        return pdf_parts

    def process_pdf_part(self, pdf_path):
        """
        Processes a single part of a PDF to extract tables.

        Args:
            pdf_path (str): Path to the PDF part.

        Returns:
            list: list of DataFrames
        """
        data_frames = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            df = pd.DataFrame(table)
                            df.replace("", None, inplace=True)  # Remove empty strings
                            data_frames.append(df)
        except Exception as e:
            print(f"⚠️ Error processing PDF part: {e}")
        finally:
            # Use os.path.exists
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    print(f"✅ Successfully deleted temporary PDF part: {pdf_path}")
                except Exception as e:
                    print(f"⚠️ Error deleting temporary PDF part: {e}")
        return data_frames if data_frames else [pd.DataFrame()]

    def extract_tables_from_pdf_parallel_processing(self, pdf_path):
        """
        Extracts tables from a PDF file using parallel processing.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            list: A list of Pandas DataFrames, or an empty list if no tables are found.
        """

        pdf_parts = self.split_pdf_in_memory(pdf_path)

        # Handle case when no PDF parts were generated
        if not pdf_parts:
            print(f"⚠️ Error: No parts generated for {pdf_path}.")
            return [pd.DataFrame()]  # Return empty DataFrame

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(self.process_pdf_part, pdf_parts))

        combined_results = []
        for result in results:
            if result:  # Ensure result is not None or empty
                combined_results.extend(result)

        return combined_results if combined_results else [pd.DataFrame()]

    def _preprocess_data_in_directory(self):
        directory_path = self.unstructured_directory_path
        logger.info(f"Preprocessing data in {directory_path}")

        elements = []
        output_file_path = os.path.join(directory_path, "output.txt")

        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if filename.endswith(".pdf"):
                    try:
                        extracted_elements = partition_pdf(file_path)
                        elements.extend([str(element) for element in extracted_elements])
                        data_frames = self.extract_tables_from_pdf_parallel_processing(pdf_path=file_path)
                        for i, table in enumerate(data_frames):
                            if not table.empty:
                                csv_path = os.path.join(directory_path, f"table_{i + 1}.csv")
                                table.to_csv(csv_path, index=False, encoding="utf-8")
                    except Exception as e:
                        logger.error(f"Error processing PDF {filename}: {e}")
                elif filename.endswith(".txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            text = infile.read()
                        preprocessed_text = self._preprocess_text(text)
                        outfile.write(preprocessed_text + "\n")
                    except Exception as e:
                        logger.error(f"Error processing txt file {filename}: {e}")
        return elements