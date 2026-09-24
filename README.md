# Zepto AI/ML Capstone Project

This repository contains my complete AI/ML capstone project. The project is divided into three modules: a data pipeline, a Titanic analytics and machine-learning pipeline, and a Zepto policy support assistant.

---

# Module 1: Data Pipeline

## What this module does

In this module, I created a small end-to-end data pipeline using the Books to Scrape website.

The process is:

1. Scrape book information from the website.
2. Clean the scraped data and convert the required columns into proper data types.
3. Convert the book prices from GBP to INR using the fixed project rate of **1 GBP = 105.50 INR**.
4. Store the cleaned data in a SQLite database.
5. Run SQL queries on the database.
6. Use pandas to read the SQL results and compare the JOIN result with `pandas.merge()`.

The scraper collects the book title, price, rating, availability and category.

## Installation

From the project root, run:

```bash
python -m pip install -r data_pipeline/requirements.txt
```

## Running the pipeline

```bash
python data_pipeline/data_pipeline.py
```

After running the program, the following files are created:

```text
books_raw.csv
books_cleaned.csv
zepto_books.db
query_outputs.txt
join_comparison.csv
```

## Data cleaning

During cleaning, I converted the price and rating into numeric values and converted the availability information into a boolean field.

If a row is missing an important field such as category, rating or availability, I remove that row instead of filling it with information that was not present in the original data. The program also displays the number of missing and removed records.

The SQLite database contains separate `categories` and `books` tables with a primary-key/foreign-key relationship.

---

# Module 2: Analytics Pipeline

## What this module does

This module uses the Titanic dataset to go through the complete analytics and machine-learning process.

I first explore and clean the data, then perform visual analysis and finally build classification and regression models.

The main steps are:

* Check the structure and missing values in the dataset
* Clean the data
* Analyze age and fare
* Look at survival rates by different groups
* Create visualizations
* Check correlations
* Train classification models
* Compare model performance
* Test different approaches for class imbalance
* Tune a Random Forest model
* Build a regression model for predicting fare
* Save the complete classification pipeline

## Running the module

Install the required packages:

```bash
python -m pip install -r analytics/requirements.txt
```

Then run:

```bash
python analytics/module2_analytics.py
```

On the first run, the Titanic dataset is loaded using:

```python
sns.load_dataset("titanic")
```

The dataset is then saved as `analytics/titanic.csv`. This allows the remaining analysis to be run using the local CSV file if the online dataset is not available.

## Files produced

The analysis produces files such as:

```text
titanic.csv
generated_report.md
classification_metrics.csv
imbalance_comparison.csv
regression_metrics.csv
best_classification_pipeline.joblib
```

The charts are saved under:

```text
analytics/charts/
```

These include the required EDA charts, correlation heatmap, confusion matrices, ROC curves, decision tree and residual plot.

## Modeling approach

For the classification models, I use a stratified train/test split before applying preprocessing.

The preprocessing is handled through a `ColumnTransformer` and `Pipeline`. This keeps imputation, encoding and scaling within the training process and prevents the test data from being used when fitting the preprocessing steps.

I also compare three approaches for class imbalance:

* Normal model
* `class_weight="balanced"`
* SMOTE applied to the training data

The Random Forest model is tuned using `GridSearchCV`.

The final saved `.joblib` file contains the complete preprocessing and model pipeline rather than only the trained estimator.

## Before submission

The generated report contains the values calculated during execution. I will review those results and make sure the written explanations accurately describe my own analysis and the actual output of the program.

---

# Module 3: Zepto Policy Support Assistant

## What this module does

This module is a small question-answering application based on eight Zepto policy documents.

The idea is to take the policy documents, create embeddings for them, store those embeddings in ChromaDB and retrieve the most relevant information when a user asks a policy-related question.

The application uses LangGraph for the flow and FastAPI to expose the `/ask` API.

The required version of the project works in mock mode, so an LLM API key is not required.

## Overall flow

```text
Zepto policy documents
        ↓
Load and split documents
        ↓
Create local embeddings
        ↓
Store embeddings in ChromaDB
        ↓
User question
        ↓
LangGraph intent classification
        ↓
 ┌───────────────────────┐
 │                       │
Policy question       General question
 │                       │
 ↓                       ↓
Retrieve relevant      Direct answer
chunks
 │
 ↓
Create response
        ↓
Pydantic validation
        ↓
FastAPI response
```

## Documents

The `docs` folder contains the eight policy documents supplied for the project:

```text
docs/
├── doc_01.txt
├── doc_02.txt
├── doc_03.txt
├── doc_04.txt
├── doc_05.txt
├── doc_06.txt
├── doc_07.txt
└── doc_08.txt
```

## Embeddings and retrieval

I use the `all-MiniLM-L6-v2` model through Sentence Transformers to create the embeddings locally.

The embeddings are stored in a ChromaDB collection called:

```text
zepto_policy_corpus
```

For a policy-related question, the application retrieves the three most relevant chunks using cosine similarity.

## LangGraph flow

The graph contains three main nodes:

```text
classify_intent
       ↓
 ┌─────┴──────────┐
 ↓                ↓
retrieve_       direct_
and_answer       answer
```

The classifier checks whether the question is related to one of the supported Zepto policy topics.

For example, words such as:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

are treated as policy-related questions in the required mock mode.

## Mock mode

The project uses `MOCK_LLM` to control whether a real LLM is used.

By default, the application runs without an external LLM:

```text
MOCK_LLM=1
```

or with the variable left unset.

In this mode, the application uses the required deterministic logic and does not need an LLM API key.

The optional real-LLM path can be enabled with:

```text
MOCK_LLM=0
```

## Response format

The API response is validated using Pydantic and contains:

```json
{
  "answer": "string",
  "sources": ["chunk IDs"],
  "confidence": 1.0
}
```

For policy questions, the `sources` field contains the retrieved document/chunk IDs. For general questions, it remains empty.

## Installation

From the `support_assistant` directory:

```bash
python -m pip install -r requirements.txt
```

## Prepare the documents

```bash
python ingest.py
```

The first time the embedding model is used, `all-MiniLM-L6-v2` may need to be downloaded.

## Test the application

```bash
python smoke_test.py
```

The smoke test includes examples for both a policy-related question and a general question.

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

The FastAPI documentation can then be opened through:

```text
http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"query\":\"What is the refund policy?\"}"
```

The actual example responses from my test run are documented in `example_responses.md`.

## Docker

The application also includes a Dockerfile.

From the `support_assistant` directory:

```bash
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

The FastAPI service will then be available on port `7860`.

## Optional real LLM

The real LLM integration is optional for this project.

If I test it locally, the API key will be supplied through an environment variable rather than stored in the repository.

For example on Windows:

```bash
set MOCK_LLM=0
set GROQ_API_KEY=your_key_here
```

API keys, `.env` files and other secrets should not be committed to GitHub.

---

# Repository Structure

```text
zepto-ai-ml-capstone/
│
├── README.md
│
├── data_pipeline/
│   ├── data_pipeline.py
│   ├── requirements.txt
│   └── ...
│
├── analytics/
│   ├── module2_analytics.py
│   ├── requirements.txt
│   ├── titanic.csv
│   ├── best_classification_pipeline.joblib
│   └── charts/
│
└── support_assistant/
    ├── docs/
    ├── main.py
    ├── rag_graph.py
    ├── prompt.py
    ├── ingest.py
    ├── smoke_test.py
    ├── example_responses.md
    ├── requirements.txt
    └── Dockerfile
```

# Final Notes

All three modules are kept in the same repository as required by the capstone instructions.

The project should be run and tested locally before submission so that the README, generated results and example responses match the actual implementation.
