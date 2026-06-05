from tavily import TavilyClient
from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
from urllib.parse import urlparse

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFERENCE_MODEL = "llama-3.3-70b-versatile"
SOURCE_TYPE_WEIGHTS = {"academic" : 2, "government" : 1.9, "organization" : 1.5, "news" : 1.4, "unknown" : 1.0, "blog" : 0.6, "social" : 0.4}
LABEL_WEIGHTS = {"supporting" : 1, "neutral" : 0, "contradicting" : -1}

client = OpenAI(api_key = GROQ_API_KEY, base_url = "https://api.groq.com/openai/v1")

st.title("AI Claim Verification Agent")
claim = st.text_area("Write your claim here:")

def llm_response(llm_prompt):
    response = client.responses.create(input = llm_prompt, model = INFERENCE_MODEL)
    return response.output_text

def ask_llm_source_type(domain):
    prompt = f"I will give you a domain and your job is to tell me the most likely source this domain refers to. Your response must be one word from these: academic, news, government, organization, social, blog, unknown. Output unknown if you are unsure about the source\nDomain: {domain}"
    response = llm_response(prompt).lower()
    return response

def get_source_type(url):
    domain = urlparse(url).netloc

    if any(substring in domain for substring in [".edu", "arxiv", "pubmed"]):
        return "academic"
    elif ".gov" in domain:
        return "government"
    else:
        response = ask_llm_source_type(domain)
        return response
    
def calculate_heuristic_score(predictions):
    heuristic_sum = 0
    total_weights = 0
    for prediction in predictions:
        heuristic_sum += prediction[0] * prediction[1] * prediction[2]
        total_weights += prediction[1] * prediction[2]
    heuristic_score = heuristic_sum / total_weights
    confidence_score = ((heuristic_score + 1) / 2)
    return confidence_score

if st.button("Verify Claim"):
    if claim:
        llm_predictions = []
        tavily_client = TavilyClient(api_key = TAVILY_API_KEY)
        tavily_response = tavily_client.search(claim, max_results = 3)
        tavily_response = tavily_response["results"]
        for response in tavily_response:
            source_weight = SOURCE_TYPE_WEIGHTS[get_source_type(response["url"])]
            content = response["content"]
            prompt = f"I am going to provide you with a claim and evidence. Your job is to see the claim and evidence and decide whether the evidence is supporting the claim, contradicting the claim, or neutral. Also give a credibility score ranging from 1 to 10 (both inclusive). \nUse this criteria to score: Does it make factual claims or just mentions opinions, Does it mention data or statistics, Does it refer or cite other sources, Is language neutral or emotional.\nYour response should be in the exact format (don't include < and >): <label score> where label can be supporting, contradicting, or neutral and score is integer number.\nClaim: {claim}\nEvidence: {content}"
            response = llm_response(prompt).lower().strip().split(' ')
            source_data = (int(LABEL_WEIGHTS[response[0]]), int(response[1]), source_weight)
            llm_predictions.append(source_data)
        confidence_score = calculate_heuristic_score(llm_predictions)
        percentage_confidence = round(confidence_score * 100, 1)
        st.write("Confidence Percentage: ", percentage_confidence)