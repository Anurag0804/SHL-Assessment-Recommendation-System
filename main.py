from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from query_functions import query_handling_using_LLM_updated  
from sentence_transformers import SentenceTransformer
import os
import torch
import google.generativeai as genai
from dotenv import load_dotenv

app = FastAPI()

# Global objects to be initialized on startup
model = None
gemini_model = None
catalog_df = None
corpus = None
corpus_embeddings = None

@app.on_event("startup")
def startup_event():
    global model, gemini_model, catalog_df, corpus, corpus_embeddings

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    print("🚀 Loading models and data...")

    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load Gemini
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")

    # Load and process catalog data
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

    catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
    corpus = catalog_df['combined'].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    print("✅ Startup complete.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Request body
class QueryRequest(BaseModel):
    query: str

# Response model
class Assessment(BaseModel):
    assessment_name: str
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]
    skills: List[str]  

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(request: QueryRequest):
    try:
        df: pd.DataFrame = query_handling_using_LLM_updated(
            request.query,
            model=model,
            gemini_model=gemini_model,
            catalog_df=catalog_df,
            corpus=corpus,
            corpus_embeddings=corpus_embeddings
        )

        if df.empty:
            raise HTTPException(status_code=404, detail="No assessments found.")

        results = []

        for _, row in df.iterrows():
            results.append({
                "assessment_name": row["Assessment Name"],
                "url": row["URL"],
                "adaptive_support": row["Adaptive/IRT"],
                "description": row["Description"],
                "duration": int(row["Duration"]),
                "remote_support": row["Remote Testing Support"],
                "test_type": row["Test Type"] if isinstance(row["Test Type"], list) else [row["Test Type"]],
                "skills": row["Skills"] if isinstance(row["Skills"], list) else [skill.strip() for skill in str(row["Skills"]).split(",")]
            })

        return {"recommended_assessments": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))