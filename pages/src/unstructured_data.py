import os, inspect, types
import streamlit as st
import traceback
import shutil
import json, os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Disable TensorFlow in transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_PYTORCH"] = "0"

from datetime import datetime
from .vstore import VectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate,PromptTemplate
import sys , re
# sys.path.insert(0,'/vendor/dependencies/crewAI')
# from vendor.dependencies.crewAI.crewai.agent import Agent
# from vendor.dependencies.crewAI.crewai.task import Task
# from vendor.dependencies.crewAI.crewai.crew import Crew
# sys.path.pop(0)
from concurrent.futures import ThreadPoolExecutor
from crewai import Agent,Task,Crew
from langchain.tools import tool

from streamlit_extras import colored_header
from pages.src.Tools.llms import get_llm
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from streamlit_extras.dataframe_explorer import dataframe_explorer
from pages.src.Tools.langroupchain_custom import LangGroupChain
from langchain.vectorstores.base import VectorStoreRetriever
#from langchain.chains.llm import LLMChain
import pdfplumber
import pandas as pd
import assemblyai as aai
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pymongo
from streamlit_option_menu import option_menu
from bson import ObjectId
from chromadb import PersistentClient  # Correct import
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings  # ✅ Use Hugging Face Embeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.chains import retrieval_qa
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_chroma import Chroma
from langchain_groq import ChatGroq    # for Groq api
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain.schema import Document
from langchain.schema.runnable import Runnable
from langchain.vectorstores.base import VectorStoreRetriever  # ✅ Correct!
from chromadb.config import Settings as ChromaSettings 
from langchain_core.runnables import RunnableMap

from .Tools.tools import (
    SEARCH_API,
    Scraper,
    Vision,
    arxiv,
    wikipedia,
    datetimee,
    YoutubeVideoTranscript
)
from textwrap import dedent
from joblib import load
from dotenv import load_dotenv

#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)


#tools initialization
@tool('Analyzer', return_direct=False)
def datanalystbotwrapper(query:str):
    '''Analyzes unstructured pdf data based on user question. User question is fed into query variable.'''
    result=st.session_state.unstructured_analyzer.datanalystbot(query=query)
    return result

@tool('Vision')
def vision(query:str):
    """This tool answers questions based on the current screen. But remember the context of the chat to answer follow up questions. Provide question to this helper agent to get answer."""
    vision=Vision(model=st.session_state.vision_model)
    return vision.vision(query=query)

@tool("Search Agent")    
def SearchAgent(query):
    """Searches internet for answers"""
    lgchain=LangGroupChain(llm=st.session_state.unstructured_analyzer._get_llm(), tools=[SEARCH_API, Scraper, datanalystbotwrapper, arxiv, wikipedia])
    return lgchain.generate_response3(query)
    

class unstructured_Analyzer:
    def __init__(self, config_data, prompt_data):

        self.config_data=config_data
        self.prompt_data=prompt_data       
        self.unstructured_directory=self.config_data['unstructured_data']
        self.image_path=self.config_data['QnA_img']
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store=VectorStore()
        st.session_state.can_upload=True
        if "messages" not in st.session_state:
            st.session_state['messages']=[]
        if 'docs_found' not in st.session_state:
            st.session_state['docs_found']=[]
        if 'internet_access' not in st.session_state:
            st.session_state['internet_access']=False
        if 'LangGroupChain' not in st.session_state:
            st.session_state['LangGroupChain']=False
        if 'extracted_tables' not in st.session_state:
            st.session_state['extracted_tables']=[]

        classification_model_path=self.config_data['Classification_models']
        st.session_state.tfidf=load(os.path.join(classification_model_path,'tfidf_pretrained.joblib'))
        st.session_state.rf=load(os.path.join(classification_model_path,'randomtree_decision_pretrained.joblib'))
        st.session_state.response_sentiment=load(os.path.join(classification_model_path,'response_sentiment.joblib'))
        

    def _upload_pdf(self):
        uploaded_files = st.file_uploader("Upload file", 
            type=["pdf",'jpg','jpeg','png'], 
            accept_multiple_files=False, 
            key=1, 
            label_visibility='hidden' )

        if uploaded_files is not None:  # ✅ Prevents AttributeError
            logging.info(f'File uploaded: {uploaded_files.name}, Type: {uploaded_files.type}')
            st.session_state['uploaded_files'] = uploaded_files  # ✅ Only store if a file is uploaded
            st.success(f"File '{uploaded_files.name}' uploaded successfully!")

        return uploaded_files  # ✅ Returns the uploaded file if available    

    def extract_tables_from_pdf(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            tables = []
            for page in pdf.pages:
                extracted_table = page.extract_table()
                if extracted_table:
                    tables.extend(extracted_table)
        
        if tables:
            return pd.DataFrame(tables[1:], columns=tables[0])  # Convert to DataFrame
        return None

    def read_extracted_table(self):
        table_path = os.path.join(self.unstructured_directory, 'extracted_table.csv')
        if os.path.exists(table_path):
            return pd.read_csv(table_path).to_dict()
        return "No extracted table found."


    def _upload_image(self):
        # for file in os.listdir(self.image_path):
        #     os.remove(os.path.join(self.image_path,file))

        if 'vision_model' not in st.session_state:
            st.session_state['vision_model']=''

        st.session_state.vision_model=st.selectbox('Select LVM',self.config_data['supported_llms']['vision_models'])

        if st.session_state.vision_model:
            uploaded_image=st.file_uploader("Upload an image",type=["png", "jpg" , "jpeg"],key=2, accept_multiple_files=False, label_visibility="hidden")
            logging.info("name :",uploaded_image)
        return uploaded_image

    def _IsGenerator(self, obj):
        return inspect.isgeneratorfunction(obj) or isinstance(obj, types.GeneratorType)
    def _decision(self,sentence):
        sentence_vectorized=st.session_state.tfidf.transform([sentence])
        prediction=st.session_state.rf.predict(sentence_vectorized)
        return prediction[0]
    def response_sentiment(self,response):
        vectorizer=load(os.path.join(self.config_data['Classification_models'],'response_sentiment_vectorizer_pretrained.joblib'))
        response_vectorized=vectorizer.transform([str(response)])
        prediction_proba=st.session_state.response_sentiment.predict_proba(response_vectorized)[0][0]
        logging.info(prediction_proba)
        return prediction_proba

  #  @st.cache_resource
    def _vstore_embeddings(self, uploaded_files=None, mongo=False, _mongo_data=None):
        if 'vectorstoreretriever' not in st.session_state:
            st.session_state['vectorstoreretriever'] = None
        if 'bm25retriever' not in st.session_state:
            st.session_state['bm25retriever'] = None

        try:
            # ✅ Clear existing Chroma vector DB if new file is uploaded
            if uploaded_files and os.path.exists("db"):
                try:
                    shutil.rmtree("db", ignore_errors=False)
                    logging.info("🧹 Old Chroma vector store directory 'db/' removed successfully.")
                except Exception as e:
                    logging.error(f"⚠️ Error deleting vector store directory: {e}")

            # ✅ Delete unnecessary files except extracted tables
            for file in os.listdir(self.unstructured_directory):
                file_path = os.path.join(self.unstructured_directory, file)
                if not file.endswith((".csv", ".json", ".txt")):
                    try:
                        os.remove(file_path)
                        logging.info(f"Deleted unnecessary file: {file}")
                    except Exception as e:
                        logging.error(f"⚠️ Failed to delete file {file}: {e}")

            if uploaded_files:
                file_type = uploaded_files.type.split('/')[1]

                # ✅ Handle PDFs
                if file_type == 'pdf':
                    pdf_path = os.path.join(self.unstructured_directory, uploaded_files.name)
                    with open(pdf_path, 'wb') as f:
                        f.write(uploaded_files.getbuffer())
                    logging.info(f'Saved PDF: {pdf_path}')

                    extracted_tables = self.extract_tables_from_pdf(pdf_path)
                    if extracted_tables is not None and not extracted_tables.empty:
                        table_path = os.path.join(self.unstructured_directory, 'extracted_table.csv')
                        extracted_tables.to_csv(table_path, index=False)
                        logging.info(f'Extracted tables saved to: {table_path}')
                    else:
                        logging.warning(f'No tables extracted from PDF: {pdf_path}')

                # ✅ Handle Audio Files
                elif file_type in ['mp3', 'mp4', 'mpeg4', 'mpeg']:
                    logging.info('Processing audio file...')
                    aai.settings.api_key = st.secrets['ASSEMBLYAI_API_KEY']['api_key']
                    with st.spinner('Collecting transcripts...'):
                        audio_dir = self.config_data['audio_dir']
                        audio_file_path = os.path.join(audio_dir, uploaded_files.name)
                        with open(audio_file_path, "wb") as f:
                            f.write(uploaded_files.getbuffer())

                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(audio_file_path)

                    transcript_path = os.path.join(self.unstructured_directory, 'transcript.txt')
                    with open(transcript_path, 'w') as f:
                        f.write(transcript.text)

                    logging.info(f'Transcript saved to: {transcript_path}')
                    for file in os.listdir(audio_dir):
                        try:
                            os.remove(os.path.join(audio_dir, file))
                            logging.info(f"Deleted audio file: {file}")
                        except Exception as e:
                            logging.error(f"⚠️ Failed to delete audio file {file}: {e}")

                # ✅ Generate embeddings
                with st.spinner('Generating Embeddings. May take some time...'):
                    store, bm25 = st.session_state.vector_store.makevectorembeddings(
                        embedding_num=st.session_state.embeddings
                    )

                    # Ensure store is always a list
                    if store is not None and not isinstance(store, list):
                        store = [store]

                    st.session_state.vectorstoreretriever = store
                    st.session_state.bm25retriever = bm25
                    logging.info(f"🔍 Returned store & bm25: {store}, {bm25}")
                    if store is None:
                        logging.error("❌ Embedding creation failed — vector store is None!")

            # ✅ Handle MongoDB input
            elif mongo:
                file_path = os.path.join(self.unstructured_directory, 'mongo_data.txt')
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(str(_mongo_data) + '\n')

                with st.spinner('Generating Embeddings. Please wait...'):
                    store, bm25 = st.session_state.vector_store.makevectorembeddings(
                        embedding_num=st.session_state.embeddings, key="mongo"
                    )

                    # Ensure store is always a list
                    if store is not None and not isinstance(store, list):
                        store = [store]

                    st.session_state.vectorstoreretriever = store
                    st.session_state.bm25retriever = bm25
                    logging.info("🔍 Returned store & bm25 (mongo):", store, bm25)
                    if store is None:
                        logging.error("❌ Embedding creation failed — vector store is None!")

            logging.info(f'✅ vectorstoreretriever: {st.session_state.vectorstoreretriever}')

            return st.session_state.vectorstoreretriever

        except Exception as e:
            logging.error(f"⚠️ Error in _vstore_embeddings: {e}")
            return None


        
    def check_for_url(self,text):
        pattern=r'https://\S+'
        matches=re.findall(pattern,text)
        if len(matches)>0:
            return True
        else:
            return False
    def _promptformatter(self):
        input_variables = ['context', 'query' , 'memory','extra_documents', 'date']
        variables = """\nQUESTION: {query},\n
        Retrieved CONTEXT: {context},\n
        Retrieved DOCS: {extra_documents},\n
        MEMORY: {memory}\n
        DATE & TIME '%Y-%m-%d %H:%M:%S' : {date}\n
        Answer:"""
        template = '\n'.join([self.prompt_data['Unstructured Prompts']['analystool']['prompt'],variables])
        # Create a new HumanMessagePromptTemplate with the modified prompt
        human_message_template = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=input_variables, template=template))

        # Create a new ChatPromptTemplate with the modified HumanMessagePromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages([human_message_template])
        # logging.info(chat_prompt_template)
        return chat_prompt_template
    
    def _get_llm(self, **kwargs):
        return get_llm(st.session_state.selected_llm,st.session_state.model_temperature, self.config_data, self.llm_category)
    
   


    def extract_multiquery_documents(self,multi_query_retriever_from_llm, query):
        try:
            with st.spinner('extracting documents (multiquery)'):
                mqdocs = multi_query_retriever_from_llm.invoke(query)
                if len(mqdocs) > 5:
                    mqdocs = mqdocs[:5]
        except Exception as e:
            mqdocs = [Document('')]
        return mqdocs
    
    def extract_ensemble_documents(self,ensemble_retriever, query):
        with st.spinner('extracting documents (ensemble retriever)'):
            ensemble_docs = ensemble_retriever.invoke(input=query)
            if len(ensemble_docs) >= 5:
                ensemble_docs = ensemble_docs[:5]
        return ensemble_docs
    
    def extract_high_similarity_documents(self, vector_store_retriever, query, similarity_threshold=0.7):
        try:
            with st.spinner('Searching for docs with high similarity threshold (0.7)'):
                # Retrieve documents based on the query
                extra_data = vector_store_retriever.get_relevant_documents(query)
                
                # Filter documents based on similarity threshold
                filtered_data = [
                    ele for ele in extra_data
                    if ele.metadata.get('similarity', 0) >= similarity_threshold
                ]
                
                # Combine the filtered content into one string
                extradata = ''.join(ele.page_content + '\n' for ele in filtered_data)
                
                if not extradata:
                    logging.warning("⚠️ No documents found with sufficient similarity.")
                
                return extradata
            
        except Exception as e:
            logging.error(f"❌ Error during document extraction: {str(e)}")
            return f"❌ Error during document extraction: {str(e)}"

    def datanalystbot(self, query: str, context=" "):
        print("🧠 datanalystbot started")

        try:
            # ✅ Get the LLM and prompt template
            llm = self._get_llm()
            prompt_template = self._promptformatter()

            # ✅ Create LLMChain from prompt
           # prompt_chain = LLMChain(prompt=prompt_template, llm=llm)
            prompt_chain = prompt_template | llm  # (RunnableSequence)


            chroma_settings = ChromaSettings(
                chroma_db_impl="duckdb+parquet",  # Ensure correct storage backends
                persist_directory="db",
                anonymized_telemetry=False, is_persistent=True
            )

            # ✅ Create the document chain
            combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)

            # ✅ Ensure Hugging Face API token
            if "huggingfacehub_api_token" not in st.session_state:
                st.session_state.huggingfacehub_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
                logging.info("🔑 Hugging Face token set")

            # ✅ Initialize Hugging Face embeddings
            if "embedding_function" not in st.session_state:
                try:
                    st.session_state.embedding_function = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True},
                    )
                    logging.info("✅ Hugging Face Embeddings initialized")
                except Exception as embed_error:
                    logging.warning(f"⚠️ Failed to initialize Hugging Face embeddings: {str(embed_error)}")
                    return f"❌ Failed to initialize embeddings: {embed_error}"

            # ✅ Ensure vector retriever is initialized correctly
            if "vectorstoreretriever" not in st.session_state or not st.session_state.vectorstoreretriever:
                try:
                    vector_embeddings_retriever = self._vstore_embeddings(
                        uploaded_files=st.session_state.get("uploaded_files", []),
                        persist_directory="db",
                        client_settings=chroma_settings
                    )[0]

                    print(f"Initialized vector embeddings retriever: {vector_embeddings_retriever}")

                    if hasattr(vector_embeddings_retriever, "as_retriever"):
                        retriever = vector_embeddings_retriever.as_retriever()
                        logging.info("✅ Converted Chroma to retriever using .as_retriever()")
                    else:
                        retriever = vector_embeddings_retriever
                        logging.warning("⚠️ Using retriever without .as_retriever()")

                    # ✅ Only set retriever if valid
                    if hasattr(retriever, "get_relevant_documents"):
                        st.session_state.vectorstoreretriever = [retriever]
                        logging.info(f"✅ Vector store retriever created: {st.session_state.vectorstoreretriever}")
                    else:
                        logging.error("❌ Retriever missing `get_relevant_documents()` method!")
                        return "❌ Error: Retriever does not implement `get_relevant_documents()`"

                except Exception as store_error:
                    logging.error(f"❌ Failed to initialize vector embeddings retriever: {str(store_error)}")
                    return f"❌ Failed to initialize vector embeddings retriever:\n{store_error}"

            # ✅ Check if vectorstoreretriever is a list after initialization
            vectorstore = st.session_state.get("vectorstoreretriever", [])

            if not isinstance(vectorstore, list):
                logging.error(f"❌ vectorstoreretriever is not a list: {type(vectorstore)}")
                return f"❌ Error: vectorstoreretriever is not a list. It is of type {type(vectorstore)}"

            # 🔍 Force conversion to retriever using .as_retriever() if needed
            retrievers = []
            for vs in vectorstore:
                try:
                    if hasattr(vs, "as_retriever"):
                        r = vs.as_retriever(search_type="similarity")
                    else:
                        r = vs  # fallback, though rare

                    if hasattr(r, "get_relevant_documents"):
                        retrievers.append(r)
                    else:
                        logging.warning(f"⚠️ Object does not have get_relevant_documents: {type(r)}")
                except Exception as e:
                    logging.error(f"❌ Failed to convert vector store to retriever: {e}")

            # ❌ If no valid retrievers found, abort
            if not retrievers:
                logging.error(f"No valid retrievers found in session state: {vectorstore}")
                print("Empty retrievers list:", retrievers)
                return "❌ Error: No valid retrievers found for EnsembleRetriever!"

            # ✅ Initialize ensemble retriever
            weights = [1.0 / len(retrievers)] * len(retrievers)
            ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
            logging.info(f"📦 Ensemble retriever initialized with {len(retrievers)} retrievers")


            # ✅ Parallel document fetching with extended timeout
            with ThreadPoolExecutor() as executor:
                mqdocs_future = executor.submit(self.extract_multiquery_documents, ensemble_retriever, query)
                ensemble_docs_future = executor.submit(self.extract_ensemble_documents, ensemble_retriever, query)
                extra_data_future = executor.submit(self.extract_high_similarity_documents, retrievers[0], query)

                try:
                    # Increase timeout to 60 seconds to handle longer processing
                    mqdocs = mqdocs_future.result(timeout=60)
                    ensemble_docs = ensemble_docs_future.result(timeout=60)
                    extradata = extra_data_future.result(timeout=60)
                except Exception as e:
                    logging.error(f"❌ Error during parallel document extraction: {str(e)}")
                    return f"❌ Error during parallel document extraction: {str(e)}"

            # Combine documents
            combinedocs = mqdocs + [Document(page_content=extradata)] + ensemble_docs

            # Chain to format input and run LLM with the prompt
            document_chain = (
                RunnableMap({
                    "context": lambda x: "\n\n".join([doc.page_content for doc in x["documents"]]),
                    "query": lambda x: x["query"],
                    "memory": lambda x: x.get("memory", ""),
                    "extra_documents": lambda x: x.get("extra_documents", ""),
                    "date": lambda x: x.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                })   |  prompt_template  | llm
            )

            # Input for the chain
            chain_input = {
                "documents": combinedocs,
                "query": query,
                "memory": None,
                "extra_documents": None,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Run it
            response = document_chain.invoke(chain_input)

            # Store sources (assuming all contributed)
            sources = combinedocs

            # Save to session and return result
            if response:
                st.session_state.docs_found = list({
                    (doc.page_content, str(doc.metadata)): doc 
                    for doc in sources
                }.values())

                # Check if the response is an instance of AIMessage
                content = response.content if hasattr(response, 'content') else "No content available."

                return content
            else:
                return "No relevant results found."

        except Exception as final_error:
            logging.error(f"❌ Unexpected error occurred in datanalystbot: {str(final_error)}")
            import traceback
            return f"❌ Unexpected error occurred:\n```\n{traceback.format_exc()}\n```"
        

    def Multimodalagent(self, query):

        Multimodal = Agent(
            role=dedent(f"{self.prompt_data['Unstructured Prompts']['crewaiagent']['role']} DATE TIME {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"),
            backstory=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['backstory']),
            verbose=True,
            allow_delegation=False,
            memory=True,
            max_iter=10,
            llm = self._get_llm(),
            goal=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['goal']),
            tools=[vision,SEARCH_API, Scraper, arxiv, wikipedia, datetimee,YoutubeVideoTranscript]
            )
        task = Task(
            description=dedent(f"user question: {query}\n\n INSTRUCTIONS : {self.prompt_data['Unstructured Prompts']['crewaiagent']['description']} \n\n {['do not use' if st.session_state.uploaded_image is not None else 'use'][0]} Vision for this question\n\n CONVERSATION HISTORY : {st.session_state.messages[::-1][0:int([3 if len(st.session_state.messages)>3 else len(st.session_state.messages)][0])]}"),
            agent=Multimodal,
            async_execution=False,
            expected_output=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['expected_output']),
            result=dedent(self.prompt_data['Unstructured Prompts']['crewaiagent']['result']),
            # human_input=True

            )
        crew = Crew(
        agents=[Multimodal],
        tasks=[task],
        verbose=0,
        )

        try:
        
            result = crew.kickoff()
            return result
        
        except Exception:

            return "Try again"

    
    
    @st.cache_data
    def show_extracted_tables_from_pdf(_self,files):
        dataframes = []
        for filename in os.listdir(_self.unstructured_directory):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(_self.unstructured_directory, filename))
                dataframes.append(df)
        return dataframes
    def fetch_mongodb_data(self, uri, database_name, collection_name):
        client = pymongo.MongoClient(uri)
        db = client[database_name]
        collection = db[collection_name]
        docs = collection.find()
        
        data = []
        for doc in docs:
            # Extract all keys dynamically from the document
            doc_data = {"_id": str(doc["_id"])}
            for key, value in doc.items():
                doc_data[key] = value
            data.append(doc_data)
        
        return data
    
    
    @st.fragment
    def mongoviewer(self):
        uri = st.text_input("MongoDB URI", "mongodb://localhost:27017/")
        database_name = st.text_input("Database Name")
        collection_name = st.text_input("Collection Name")

        if st.button("Fetch Data and generate embeddings"):
            if uri and database_name and collection_name:
                    data = self.fetch_mongodb_data(uri, database_name, collection_name)
                    logging.info('generating', type(data))

                    vstore=self._vstore_embeddings(mongo=True,_mongo_data=data)
                    st.session_state.vectorstoreretriever=vstore
                    st.success(f'Embeddings generated {st.session_state.vectorstoreretriever}')
                    if data:
                        st.subheader(f"Number of docs is {len(data)}. Here is preview!")
                        for doc in [data if len(data)<20 else data[0]]:
                            st.write(doc)
                    
                    
                    else:
                        st.write("No documents found.")
                        
                        
            else:
                st.warning("Please enter all the required credentials.")


    def sidebarcomponents(self):
        if st.session_state.internet_access:
            with st.sidebar.title('Analyze images'):
                    with st.expander('Upload Images'):
                        st.session_state.uploaded_image=self._upload_image()

                        if st.session_state.uploaded_image is not None and st.session_state.uploaded_image!=[]:

                            logging.info('image uploader')
                            for ele in os.listdir(self.image_path):
                                os.remove(os.path.join(self.image_path,ele))

                            with open(os.path.join(self.image_path,(st.session_state.uploaded_image).name), 'wb') as file:
                                file.write(st.session_state.uploaded_image.read())

        if st.session_state.vectorstoreretriever is not None:
            with st.sidebar.title('Description'):
                    if 'description' not in st.session_state:
                        st.session_state.description=' '
                    
                    with st.popover('Describe',help='provide description about the data provided to aid better retrieval.', use_container_width=True):
                        st.session_state.description=st.text_area('Describe Data')

    def generateresponse(self, prompt):
        # Ensure required session state keys exist
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("internet_access", False)
        st.session_state.setdefault("description", "No description provided.")
        st.session_state.setdefault("uploaded_image", None)
        st.session_state.setdefault("LangGroupChain", False)
        st.session_state.setdefault("docs_found", [])

        # Step 1: Decision engine
        predict = self._decision(prompt)  # Make sure _decision() is implemented
        if self.check_for_url(prompt):    # Returns True if URL detected
            predict = "search"

        # Step 2: Log user prompt
        st.session_state.messages.append({'role': 'user', 'question': prompt})
        message = "Sorry, I couldn't understand the prompt."

        # Step 3: Internet-enabled response path
        if st.session_state.internet_access:
            st.session_state.docs_found = []  # Reset docs

            if 'analysis' in predict:
                with st.spinner('Analyzing your data...'):
                    message = self.datanalystbot(prompt, st.session_state.description)

                    # If response seems meaningful, optionally do deeper chain processing
                    if message and self.response_sentiment(message) > 0.40:
                        with st.spinner('Expanding search...'):
                            if st.session_state.LangGroupChain:
                                message = SearchAgent(prompt)  # Ensure this is globally available
                            else:
                                message = self.Multimodalagent(prompt)  # Make sure this is implemented

            elif 'vision' in predict and st.session_state.uploaded_image is not None:
                st.image(st.session_state.uploaded_image, caption="Uploaded Image")  # Display uploaded image
                with st.spinner('Analyzing image...'):
                    message = vision(prompt)  # Ensure global vision() function is defined

            else:
                with st.spinner('Searching online...'):
                    if st.session_state.LangGroupChain:
                        message = SearchAgent(prompt)
                    else:
                        message = self.Multimodalagent(prompt)

        # Step 4: Offline response path
        else:
            with st.spinner('Generating response...'):
                message = self.datanalystbot(prompt, st.session_state.description)

        logging.info(f"LLM response: {message}")
        return message



    def workflow(self):
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "docs_found" not in st.session_state:
            st.session_state.docs_found = None
        if "vectorstoreretriever" not in st.session_state:
            st.session_state.vectorstoreretriever = None

        colored_header.colored_header("Supported LLMs", "choose your llm", color_name='blue-green-90')
        
        self.llm_category = st.selectbox(
            label='choose llm category',
            options=self.config_data["llm_category"],
            label_visibility='collapsed'
        )
        
        # Select LLMs based on category
        if self.llm_category == 'gemini models':
            self.supported_llms = self.config_data["supported_llms"]['gemini_llms']
        elif self.llm_category == 'ollama models':
            self.supported_llms = self.config_data["supported_llms"]['opensource_llms']
        elif self.llm_category == 'huggingface models':
            self.supported_llms = self.config_data["supported_llms"]['huggingface_llms']
        elif self.llm_category == 'openai models':
            self.supported_llms = self.config_data["supported_llms"]['openai_llms']
        elif self.llm_category == 'groq models':
            self.supported_llms = self.config_data["supported_llms"]['groq_llms']
        elif self.llm_category == 'antrophic models':
            self.supported_llms = self.config_data["supported_llms"]['antrophic_llms']

        # UI: LLM + temperature
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.selected_llm = st.selectbox("LLMS", self.supported_llms)
        with col2:
            st.session_state.model_temperature = st.slider(
                'Model Temperatures', min_value=0.1, max_value=1.0, value=0.5, step=0.01
            )

        # Toggle internet access and LangGroupChain
        st.session_state.internet_access = st.toggle(
            'Activate Internet Access',
            help="This will enable llm to search internet for queries"
        )
        
        st.session_state.LangGroupChain = st.toggle(
            'Activate LangGroupChain Experimental?',
            help="Works with internet access. Experimental chain that breaks down a problem into graph..."
        )

        # Document source selection
        select_upload_option = option_menu(None, ['Upload document', 'MONGO DB'], orientation="horizontal")
        
        files = None  # Initialize

        if select_upload_option == 'MONGO DB':
            self.mongoviewer()

        elif select_upload_option == 'Upload document':
            files = self._upload_pdf()
            st.session_state.vectorstoreretriever = self._vstore_embeddings(uploaded_files=files)

        # Show vector store info (debug)
        st.write(st.session_state.vectorstoreretriever)

        if st.session_state.vectorstoreretriever is not None and files is not None:
            # Show extracted tables
            with st.expander('Extracted Tables From Docs'):
                tables = self.show_extracted_tables_from_pdf(files)
                if tables:
                    if len(tables) > 10 or len(tables[0]) > 100:
                        st.warning('Large number of tables detected. Consider analyzing in the Structured section.')
                    st.warning('Note: You can use these tables for analysis in the structured section.')

                    for i, table in enumerate(tables):
                        st.subheader(f'Table {i+1}', divider=True)
                        st.data_editor(table)
                else:
                    st.warning('No Tables Found')

            # Display conversation
            for message in st.session_state.messages:
                with st.chat_message(message['role']):
                    if message['role'] == 'user':
                        st.write(message['question'], unsafe_allow_html=True)
                    elif message['role'] == 'assistant':
                        if self._IsGenerator(message['reply']):
                            st.write_stream(message['reply'])
                        elif isinstance(message['reply'], pd.DataFrame):
                            st.dataframe(message['reply'])
                        else:
                            st.write(message['reply'], unsafe_allow_html=True)

            # Show retrieved documents
        #    if st.session_state.docs_found:
        #        with st.expander('retrieved docs'):
         #           st.write(st.session_state.docs_found, unsafe_allow_html=True)

            # Handle chat input
            if prompt := st.chat_input('Ask questions', key='data_chat'):
                logging.info(prompt)
                message = self.generateresponse(prompt=prompt)
                st.session_state.messages.append({'role': 'assistant', 'reply': message})
                st.rerun()



    def run(self):
        self.workflow()
        self.sidebarcomponents()





            

            

                

            
