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
    if total_weights == 0:
        return 0.5
    heuristic_score = heuristic_sum / total_weights
    confidence_score = ((heuristic_score + 1) / 2)
    return confidence_score

if st.button("Verify Claim"):
    if claim:
        prompt = f"I will provide you a claim. Your job is to determine if there is ambiguity in it or not. If there is no ambiguity just output the word none, else output the unambiguous claim (only new claim no extra words)\nClaim: {claim}"
        response = llm_response(prompt)
        if response != "none":
            claim = response
            st.write(f"Reframed Claim: {claim}")
        sources_info = []
        sources_data = []
        tavily_client = TavilyClient(api_key = TAVILY_API_KEY)
        tavily_responses = tavily_client.search(claim, max_results = 3, search_depth = "advanced", exclude_domains = ["youtube.com", "pinterest.com", "reddit.com", "instagram.com", "facebook.com", "tiktok.com", "x.com"])
        tavily_responses = tavily_responses["results"]
        for tavily_response in tavily_responses:
            source_weight = SOURCE_TYPE_WEIGHTS[get_source_type(tavily_response["url"])]
            content = tavily_response["content"]
            prompt = f"I am going to provide you with a claim and evidence. Your job is to see the claim and evidence and decide whether the evidence is supporting the claim, contradicting the claim, or neutral. Also give a credibility score ranging from 1 to 10 (both inclusive). \nUse this criteria to score: Does it make factual claims or just mentions opinions, Does it mention data or statistics, Does it refer or cite other sources, Is language neutral or emotional.\nYour response should be in the exact format (don't include < and >): <label score> where label can be supporting, contradicting, or neutral and score is integer number.\nClaim: {claim}\nEvidence: {content}"
            try:
                response = llm_response(prompt).lower().strip().split(' ')
                source_data = (int(LABEL_WEIGHTS[response[0]]), int(response[1]), source_weight)
                sources_data.append(source_data)
                sources_info.append((source_data, response[0], tavily_response["url"]))
            except:
                pass
        confidence_score = calculate_heuristic_score(sources_data)
        percentage_confidence = round(confidence_score * 100, 1)
        st.write(f"Confidence Percentage: {percentage_confidence}%")
        index = 1
        for data in sources_info:
            st.write(f"Source {index}:  \nLabel: {data[1]}  \nCredibility Score: {data[0][1]}  \nURL: {data[2]}  \n")
            index += 1
        if percentage_confidence >= 50:
            confidence_label = "true"
        else:
            confidence_label = "false"
        prompt = f"I will give you a Claim and tell whether it is true or false. Your job is to give a small (max 2 line) explanation for why the claim is true or false.\nClaim: {claim}\nLabel: {confidence_label}"
        response = llm_response(prompt)
        st.write(f"Explanation:  \n{response}")