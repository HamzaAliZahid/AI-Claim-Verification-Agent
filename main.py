from tavily import TavilyClient
from dotenv import load_dotenv
import os
import streamlit as st
from google import genai
from urllib.parse import urlparse

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"

gemini_client = genai.Client(api_key = GEMINI_API_KEY)

st.title("AI Claim Verification Agent")
claim = st.text_area("Write your claim here:")

def gemini_response(prompt):
    response = gemini_client.models.generate_content(model = GEMINI_MODEL, contents = gemini_prompt).text
    return response

def ask_gemini_source_type(domain):
    gemini_prompt = f"I will give you a domain and your job is to tell me the most likely source this domain refers to. Your response must be one word from these: academic, news, government, organization, social, blog, unknown. Output unknown if you are unsure about the source\nDomain: {domain}"
    response = gemini_response(gemini_prompt).lower()
    return response

def get_source_type(url):
    domain = urlparse(url).netloc

    if any(substring in domain for substring in [".edu", "arxiv", "pubmed"]):
        return "academic"
    elif ".gov" in domain:
        return "government"
    else:
        response = ask_gemini_source_type(domain)
        return response

if st.button("Verify Claim"):
    if claim:
        tavily_client = TavilyClient(api_key = TAVILY_API_KEY)
        tavily_response = tavily_client.search(claim)
        st.write(tavily_response)
        tavily_response = tavily_response["results"]
        for response in tavily_response:
            content = response["content"]
            gemini_prompt = f"I am going to provide you with a claim and evidence. Your job is to see the claim and evidence and decide whether the evidence is supporting the claim, contradicting the claim, or neutral. You can only give one word answer from 3 words: claim, contradicting, neutral\nClaim: {claim}\nEvidence: {content}"
            response = gemini_response(gemini_prompt).lower()
            st.write(gemini_response)