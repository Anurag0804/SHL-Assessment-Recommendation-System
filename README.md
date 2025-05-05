
# SHL Assessment Recommendation System

This project is an AI-powered recommendation system designed to suggest the most relevant SHL assessments based on user queries, job descriptions, or unstructured input. By utilizing advanced NLP techniques and Large Language Models (LLMs), it delivers accurate, context-aware results via a user-friendly frontend.

**Try it out @** [Live Demo](https://shlassessmentsrecommendationsystem.streamlit.app/)

## Features

* Recommend SHL assessments based on:

  * Job descriptions
  * Unstructured URLs or text input
  * Custom user queries
* NLP-based semantic similarity matching using Sentence-BERT
* Contextual feature extraction via Gemini 1.5 Pro (LLM)
* Cosine similarity for ranking and scoring
* Top recommendations with relevance filtering
* Streamlit frontend for interactive user experience

## Tech Stack

* **Natural Language Processing**:

  * Sentence-BERT for query and dataset embeddings
  * Cosine similarity for assessment ranking
* **LLM Integration**:

  * Gemini 1.5 Pro for extracting structured features from unstructured input
  * Post-processing based on constraints (e.g., duration, skill match)
* **Frontend**:

  * Streamlit for a user-friendly interface

## How It Works

### Data Preparation:

* A mock dataset of 50 SHL-like assessments is used.
* Each entry includes assessment name, URL, duration, test type, skills, description, and adaptive/IRT support.
* Data is processed into a "combined" column for embedding.

### NLP Embedding and Retrieval:

* Sentence-BERT transforms dataset entries and input queries into vector embeddings.
* Cosine similarity is used to identify the most relevant assessments.

### LLM Enhancement (Gemini 1.5 Pro):

* Accepts job descriptions, URLs, or unstructured queries.
* Extracts structured features like job role, required skills, duration, etc.
* Re-filters results for improved accuracy.

### Evaluation:

* Performance metrics include Recall\@5 and MAP\@5.
* Hybrid model (NLP + LLM) outperforms NLP-only baseline.

### Streamlit Interface:

* Direct query input from users.
* Displays top recommended assessments with details.

## Performance

* **NLP Model**: Recall\@5 = 0.85, MAP\@5 = 0.71
* **NLP + LLM Model**: Recall\@5 = 1.0, MAP\@5 = 1.0

## Test Cases

* **Example 1**: "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
* **Example 2**: "Looking to hire mid-level professionals who are proficient in Python, SQL, and JavaScript. Need an assessment package that can test all skills with a max duration of 60 minutes."
* **Example 3**: "I am hiring for an analyst and want to screen candidates using cognitive and personality tests within 45 minutes."
* **Example 4**: "Job Listing: [Research Engineer - AI](https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in)"
* **Example 5**: "Want to assess communication and teamwork skills in under 30 minutes."

