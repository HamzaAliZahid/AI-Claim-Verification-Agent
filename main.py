from tavily import TavilyClient
from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
from urllib.parse import urlparse
from typing import TypedDict
from langgraph.graph import START, StateGraph, END

class graph_state(TypedDict):
    claim : str
    search_results : list
    retry_count : int
    final_result : str
    sources_data : list
    sources_info : list
    current_confidence_percentage : float
    current_confidence_label : str

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

def get_source_type(url):
    domain = urlparse(url).netloc
    if any(substring in domain for substring in [".edu", "arxiv", "pubmed", "acm.org", "ieee"]):
        return "academic"
    elif any(substring in domain for substring in [".gov", ".gov."]):
        return "government"
    elif any(substring in domain for substring in ["bbc", "cnn", "reuters", "apnews", "aljazeera", "nytimes", "washingtonpost", "theguardian"]):
        return "news"
    elif any(substring in domain for substring in ["twitter", "x.com", "reddit", "facebook", "instagram", "tiktok"]):
        return "social"
    elif any(substring in domain for substring in ["wikipedia", "britannica"]):
        return "organization"
    elif any(substring in domain for substring in ["medium", "blog", "substack"]):
        return "blog"
    else:
        return "unknown"
    
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

def clarify_claim(state):
    prompt = f"I will provide you a claim. Check only if the claim is ambiguous (missing key context like location, time, or subject that makes it unclear what is being claimed). If ambiguous, output a more specific version. If not ambiguous, output exactly 'none'. Do not fact-check, do not judge if the claim is true or false, do not add explanation. Make sure that the clarified claim is different from the claim I give you as input. Only output the clarified claim or 'none'.\nClaim: {state['claim']}"
    response = llm_response(prompt)
    if response != "none" and response.lower().strip() != state['claim'].lower().strip():
        st.write(f"Clarified Claim: {response}")
        return {"claim" : response}
    return {}

def search_tavily(state):
    tavily_client = TavilyClient(api_key = TAVILY_API_KEY)
    tavily_responses = tavily_client.search(state["claim"], max_results = 3, search_depth = "advanced", exclude_domains = ["youtube.com", "pinterest.com", "reddit.com", "instagram.com", "facebook.com", "tiktok.com", "x.com"])
    tavily_responses = tavily_responses["results"]
    return {"search_results" : tavily_responses}

def calculate_confidence(state):
    confidence_score = calculate_heuristic_score(state["sources_data"])
    percentage_confidence = round(confidence_score * 100, 1)
    st.write(f"Confidence Percentage: {percentage_confidence}%")
    index = 1
    for data in state["sources_info"]:
        st.write(f"Source {index}:  \nLabel: {data[1]}  \nCredibility Score: {data[0][1]}/10  \nURL: {data[2]}  \nSource Type: {data[3]}  \n")
        index += 1
    if percentage_confidence >= 50:
        confidence_label = "true"
    else:
        confidence_label = "false"
    return {"current_confidence_percentage" : percentage_confidence, "current_confidence_label" : confidence_label}

def get_sources_info(state):
    sources_data = []
    sources_info = []
    for tavily_response in state["search_results"]:
        source_type = get_source_type(tavily_response["url"])
        source_weight = SOURCE_TYPE_WEIGHTS[source_type]
        content = tavily_response["content"]
        prompt = f"I am going to provide you with a claim and evidence. Your job is to see the claim and evidence and decide whether the evidence is supporting the claim, contradicting the claim, or neutral. Also give a credibility score ranging from 1 to 10 (both inclusive). \nUse this criteria to score: Does it make factual claims or just mentions opinions, Does it mention data or statistics, Does it refer or cite other sources, Is language neutral or emotional.\nYour response should be in the exact format (don't include < and >): <label score> where label can be supporting, contradicting, or neutral and score is integer number.\nClaim: {state['claim']}\nEvidence: {content}"
        try:
            response = llm_response(prompt).lower().strip().split(' ')
            source_data = (int(LABEL_WEIGHTS[response[0]]), int(response[1]), source_weight)
            sources_data.append(source_data)
            sources_info.append((source_data, response[0], tavily_response["url"], source_type))
        except:
            pass
    return {"sources_data" : sources_data, "sources_info" : sources_info}

def final_verdict_explanation(state):
    prompt = f"I will give you a Claim and tell whether it is true or false. Your job is to give a small (max 2 line) explanation for why the claim is true or false.\nClaim: {state['claim']}\nLabel: {state['current_confidence_label']}"
    response = llm_response(prompt)
    st.write(f"Explanation:  \n{response}")
    return {}
    
def agent_pipeline(state):
    pipeline = StateGraph(graph_state)
    pipeline.add_node("clarify", clarify_claim)
    pipeline.add_node("search", search_tavily)
    pipeline.add_node("scores", get_sources_info)
    pipeline.add_node("confidence", calculate_confidence)
    pipeline.add_node("verdict", final_verdict_explanation)

    pipeline.add_edge(START, "clarify")
    pipeline.add_edge("clarify", "search")
    pipeline.add_edge("search", "scores")
    pipeline.add_edge("scores", "confidence")
    pipeline.add_edge("confidence", "verdict")
    pipeline.add_edge("verdict", END)

    graph = pipeline.compile()
    final_state = graph.invoke(state)
    return final_state

if st.button("Verify Claim"):
    if claim:
        pipeline_state : graph_state = {"claim" : claim, "search_results" : [], "retry_count" : 0, "final_result" : "", "sources_data" : [], "sources_info" : [], "current_confidence_percentage" : 0.0, "current_confidence_label" : ""}
        agent_pipeline(pipeline_state)
        