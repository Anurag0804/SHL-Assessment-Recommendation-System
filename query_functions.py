import json
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer,util
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import os

catalog_df = pd.read_csv("SHL_catalog.csv")

def combine_row(row):
    parts = [
        str(row["Assessment Name"]),
        str(row["Duration"]),
        str(row["Remote Testing Support"]),
        str(row["Adaptive/IRT"]),
        str(row["Test Type"]),
        str(row["Skills"]),
        str(row["Description"]),
    ]
    return ' '.join(parts)

catalog_df['combined'] = catalog_df.apply(combine_row,axis=1)

corpus = catalog_df['combined'].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = model.encode(corpus,convert_to_tensor=True)

def extract_url_from_text(text):
    match = re.search(r'(https?://[^\s,]+)', text)
    if match:
        return match.group(1)
    return None

def extract_text_from_url(url):
    try:
        response = requests.get(url,headers={'User-Agent':"Mozilla/5.0"})
        soup = BeautifulSoup(response.text,'html.parser')
        return ' '.join(soup.get_text().split())
    except Exception as e:
        return f"Error:{e}"

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def extract_features_with_llm(user_query):
    prompt = f"""
You are an intelligent assistant helping to recommend SHL assessments.

The input below may be:
1. A natural language query describing assessment needs (e.g., "Need a Python test under 60 minutes").
2. A job description (JD) pasted directly.
3. A job description URL (already converted into text outside this function).
4. A combination of user query + JD.

Your task is to extract and summarize key hiring features from the input. Look for and include the following **if available**:

- Job Title  
- Duration of Test  
- Remote Testing Support (Yes/No)  
- Adaptive/IRT Format (Yes/No)  
- Test Type  
- Skills Required  
- Any other relevant hiring context

Format your response as a **single line** like this:

`<Job Title> <Duration> <Remote Support> <Adaptive> <Test Type> <Skills> <Other Info>`

Skip any fields not mentioned — do not include placeholders or "N/A".

---
Input:
{user_query}

Only return the final, clean sentence — no explanations.
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def find_assessments(user_query,k=5):
    query_embedding = model.encode(user_query, convert_to_tensor = True)
    cosine_scores = util.cos_sim(query_embedding,corpus_embeddings)[0]
    top_k = min(k,len(corpus))
    top_results = torch.topk(cosine_scores,k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        result = {
            "Assessment Name": catalog_df.iloc[idx]['Assessment Name'],
            "Skills": catalog_df.iloc[idx]['Skills'],
            "Test Type": catalog_df.iloc[idx]['Test Type'],
            "Description": catalog_df.iloc[idx]['Description'],
            "Remote Testing Support": catalog_df.iloc[idx]['Remote Testing Support'],
            "Adaptive/IRT": catalog_df.iloc[idx]['Adaptive/IRT'],
            "Duration": catalog_df.iloc[idx]['Duration'],
            "URL": catalog_df.iloc[idx]['URL'],
            "Score": round(score.item(), 4)
        }
        results.append(result)
    return results

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def filter_relevant_assessments_with_llm(user_query, top_results):
    prompt = f"""
You are helping to refine assessment recommendations based on user needs.

A user has entered the following query:
"{user_query}"

You are given 10 or less assessments retrieved using semantic similarity. 
Your task is to go through each assessment and determine if it truly matches the user’s intent, based on the following:
- Duration match (e.g., if the user wants "< 40 mins", exclude longer ones)
- Skills match (e.g., user wants "Python" but test is on "Excel", reject it)
- Remote support, Adaptive format, Test type, or any clearly stated requirement
- Ignore irrelevant matches, even if score is high

Return only the assessments that are **highly relevant** to the query. 
Use your understanding of language and hiring to filter smartly. But you have to return something atleast 1 assessment.
You have to return minimum 1 assessment and maximum 10(only relevant ones). You cannot return empty json.

Respond in clean JSON format:
[
  {{
    "Assessment Name": "...",
    "Skills": "...",
    "Test Type": "...",
    "Description": "...",
    "Remote Testing Support": "...",
    "Adaptive/IRT": "...",
    "Duration": "... mins",
    "URL": "...",
    "Score": ...
  }},
  ...
]

---
Assessments:
{top_results}
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def query_handling_using_LLM_updated(query, model = model , gemini_model = gemini_model, catalog_df = catalog_df, corpus = corpus, corpus_embeddings = corpus_embeddings):
    url = extract_url_from_text(query)

    if url:
        extracted_text = extract_text_from_url(url)
        query += " " + extracted_text

    user_query = extract_features_with_llm(query)

    top_results = find_assessments(user_query, k=10)

    top_json = json.dumps(top_results, indent=2, default=convert_numpy)

    filtered_output = filter_relevant_assessments_with_llm(user_query, top_json)

    # Check for empty response
    if not filtered_output or not filtered_output.strip():
        print("Empty response from LLM.")
        return pd.DataFrame()

    # Try to extract valid JSON from the output using regex
    try:
        match = re.search(r"\[.*\]", filtered_output, re.DOTALL)
        if match:
            json_str = match.group()
            filtered_results = json.loads(json_str)
        else:
            print("⚠️ No valid JSON array found in the response:")
            print(filtered_output)
            return pd.DataFrame()
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Raw output was:\n", filtered_output)
        return pd.DataFrame()

    # Convert to DataFrame
    if not filtered_results:
        return pd.DataFrame()
    else:
        try:
            df = pd.DataFrame(filtered_results)
            print("Returning DataFrame:\n", df.head())
            return df
        except Exception as e:
            print("Error creating DataFrame:", e)
            return pd.DataFrame()