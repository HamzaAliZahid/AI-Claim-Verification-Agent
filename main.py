from tavily import TavilyClient
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.title("AI Claim Verification Agent")
claim = st.text_area("Write your claim here:")

if claim:
    if st.button("Verify Claim"):
        tavily_client = TavilyClient(api_key = TAVILY_API_KEY)
        tavily_response = tavily_client.search(claim)
        tavily_response = tavily_response["results"]