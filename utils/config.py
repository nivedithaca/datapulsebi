import os
from dotenv import load_dotenv

load_dotenv()

def get_groq_api_key() -> str:
    """Get Groq API key — tries st.secrets first, then .env / environment variable."""
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key and key != "your_groq_api_key_here":
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")
