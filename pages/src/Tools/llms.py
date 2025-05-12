from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.llms import huggingface_hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM

from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms.anthropic import Anthropic
import streamlit as st
import asyncio

def ensure_asyncio_event_loop():
    """Ensures an asyncio event loop exists for running async operations."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
def get_llm(llm_name, temperature, config_data, llm_category):
    """Initializes and returns an LLM model based on the specified category."""
    try:
        ensure_asyncio_event_loop()

        # Get API keys safely
        google_api_key = st.session_state.get("google_gemini_api")
        openai_api_key = st.session_state.get("openai_api_key")
        huggingfacehub_api_token = st.session_state.get("huggingfacehub_api_token")
        groq_api_key = st.session_state.get("groq_api_key")
        anthropic_api_key = st.session_state.get("anthropic_api_key")

        # Initialize the correct LLM
        if "gemini" in llm_category:
            if not google_api_key:
                raise ValueError("Google Gemini API key is missing.")
            llm = ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                model=llm_name,
                temperature=temperature
            )
        
        elif "huggingface" in llm_category:
            if not huggingfacehub_api_token:
                raise ValueError("Hugging Face API token is missing.")
            try:
                llm = huggingface_hub(
                    repo_id=llm_name,
                    task="text-generation",  # Hugging Face models require this
                    huggingfacehub_api_token=huggingfacehub_api_token,
                    temperature=temperature
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Hugging Face model: {e}")

        elif "openai" in llm_category:
            if not openai_api_key:
                raise ValueError("OpenAI API key is missing.")
            llm = ChatOpenAI(
                model=llm_name,
                task="text-generation",  # OpenAI models require this
                temperature=temperature,
                api_key=openai_api_key
            )

        elif "groq" in llm_category:
            if not groq_api_key:
                raise ValueError("Groq API key is missing.")
            llm = ChatGroq(
                api_key=groq_api_key,
                temperature=temperature,
                model=llm_name # No 'task' argument here
            )

        elif "ollama" in llm_category:
            llm = Ollama(
                model=llm_name,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported LLM category: {llm_category}")

        return llm

    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None
