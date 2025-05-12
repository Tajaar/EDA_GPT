import streamlit as st
import os

import streamlit as st

def initialize_secrets():
    
    st.session_state.tavily_api_key = st.secrets.get('TAVILY_API_KEY', "")
    st.session_state.bing_api_key = st.secrets.get('BING_API_KEY', "")
    st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    st.session_state.google_gemini_api = st.secrets.get("GOOGLE_GEMINI_API", "")
    st.session_state.huggingfacehub_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
    st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY", "")


    # Check and print missing keys (useful for debugging)
   # missing_keys = [key for key, value in st.session_state.items() if "api_key" in key or "api_token" in key and not value]
  #  if missing_keys:
   #     st.warning(f"⚠️ Warning: Missing API keys: {', '.join(missing_keys)}")



def initialize_states():
    
    vars = [
        'embeddings', 'vectorstoreretriever', 'loaded_vstore', 'uploaded_files',
        "current_page", "huggingfacehub_api_token", "google_gemini_api",
        "openai_api_key", "groq_api_key"
    ]

    for var in vars:
        if var not in st.session_state:
            if var in vars[5:]:  # API keys (stored as simple strings)
                st.session_state[var] = st.secrets.get(var.upper(), "")
            elif var == 'current_page':
                st.session_state[var] = "INSTRUCTIONS"
            else:  # Other session variables
                st.session_state[var] = None

