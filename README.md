# AI-Powered Insurance Policy Q&A 🤖📄

### Retrieval-Augmented Generation (RAG) System for Intelligent Document Question Answering

An end-to-end **Retrieval-Augmented Generation (RAG)** system that converts complex insurance policy documents into a searchable knowledge base and uses semantic retrieval + Large Language Models (LLMs) to answer natural-language questions accurately from the source document.

The project was developed to solve a practical problem: **insurance policy documents are long, unstructured, and difficult to search manually.** Instead of relying on keyword-based search, the system retrieves semantically relevant sections of the document and provides concise answers grounded in the retrieved information.

---

## 🎯 Problem Statement

Insurance policy documents contain hundreds of clauses, exclusions, conditions, coverage limits, and claim requirements.

Finding a specific answer such as:

* Who is eligible for coverage?
* What are the base covers?
* What documents are required for a claim?
* What is the entry age limit?
* What happens in case of baggage loss?
* What are the conditions for trip cancellation?

can require manually searching through lengthy documents.

### Objective

Build an AI-powered question-answering pipeline that can:

1. Accept an insurance policy document.
2. Extract structured text from the document.
3. Split the document into meaningful chunks.
4. Create a semantic vector index.
5. Retrieve the most relevant information for a user question.
6. Generate a concise answer using an LLM.
7. Return the answer through an API.

---

## 🧠 How the RAG Pipeline Works

```text
              Insurance Policy PDF
                       │
                       ↓
              Document Extraction
                       │
                  LlamaParse
                       │
                       ↓
               Text Preprocessing
                       │
                       ↓
                 Text Chunking
                       │
                  Chonkie
                       │
                       ↓
             Semantic Vector Index
                       │
                    Pinecone
                       │
                       ↓
                User Question
                       │
                       ↓
             Semantic Retrieval
                       │
                       ↓
              Relevant Text Chunks
                       │
                       ↓
                  LLM Prompt
                       │
                     Groq
                       │
                       ↓
              Grounded Answer
```

The key idea is that the LLM does **not need to process the entire document for every question**.

Instead, the system first retrieves the most relevant document sections and then passes those sections to the LLM as context.

---

## 🔬 Data & NLP Pipeline

### 1. Document Extraction

The system uses **LlamaParse** to extract text from PDF documents and convert the content into a structured Markdown representation.

This allows the pipeline to work with lengthy and complex documents while preserving useful document structure.

### 2. Text Chunking

Large documents cannot efficiently be passed directly to an LLM.

The extracted document is therefore divided into smaller overlapping chunks using **Chonkie**.

The chunking strategy uses:

* Sentence-aware chunking
* Configurable chunk size
* Chunk overlap
* Context refinement

This helps preserve context across chunk boundaries.

### 3. Semantic Search

Instead of matching exact keywords, the system uses **semantic retrieval**.

Document chunks are indexed in **Pinecone**, allowing the system to retrieve passages that are semantically related to the user's question.

For example:

```text
Question:
"What documents are needed for hospitalization claims?"

                    ↓

Semantic Search

                    ↓

Relevant policy sections

                    ↓

LLM Context

                    ↓

Final Answer
```

### 4. Retrieval-Augmented Generation

The retrieved document sections are inserted into a structured prompt and passed to the LLM.

The generation process is designed to:

* Answer directly
* Use information from retrieved sources
* Avoid unnecessary information
* State clearly when information is unavailable
* Keep responses concise

This reduces the risk of the LLM generating information that is unrelated to the policy document.

---

## 🤖 LLM Integration

The project integrates **Groq's LLM API** for answer generation.

The current implementation uses:

```text
moonshotai/kimi-k2-instruct
```

The LLM receives:

```text
User Question
      +
Retrieved Document Context
      ↓
   LLM Prompt
      ↓
Grounded Answer
```

The system is therefore based on **Retrieval-Augmented Generation rather than standalone LLM prompting**.

---

## 🔎 Example

### Question

> What is the entry age limit for insured members?

### Retrieved Answer

> The entry age limit for insured members is **3 months to 90 years (completed age)**.

Another example:

### Question

> What documents are required to file a hospitalization claim?

The system retrieves the relevant policy sections and generates a structured response containing items such as:

* Claim form
* Travel ticket / boarding pass
* Medical reports
* Discharge summary
* Hospital bills and receipts
* Medical prescriptions
* FIR / MLC report where applicable

The answer is generated from the retrieved policy content rather than from general internet knowledge.

---

## 🏗️ System Architecture

```text
                         ┌───────────────────┐
                         │   Policy Document │
                         └─────────┬─────────┘
                                   │
                                   ↓
                         ┌───────────────────┐
                         │   LlamaParse      │
                         │ Document Parsing  │
                         └─────────┬─────────┘
                                   │
                                   ↓
                         ┌───────────────────┐
                         │     Chonkie       │
                         │   Text Chunking   │
                         └─────────┬─────────┘
                                   │
                                   ↓
                         ┌───────────────────┐
                         │     Pinecone      │
                         │  Vector Database  │
                         └─────────┬─────────┘
                                   │
                            Semantic Search
                                   │
                                   ↓
                         ┌───────────────────┐
                         │ Relevant Chunks   │
                         └─────────┬─────────┘
                                   │
                                   ↓
                         ┌───────────────────┐
                         │       Groq        │
                         │       LLM         │
                         └─────────┬─────────┘
                                   │
                                   ↓
                         ┌───────────────────┐
                         │  Final Answer     │
                         └───────────────────┘
```

---

## 📁 Project Structure

```text
bajaj0909/
│
├── main.py
│   └── FastAPI application and RAG pipeline orchestration
│
├── data_extraction.py
│   └── PDF/document extraction using LlamaParse
│
├── chunks.py
│   └── Sentence-based document chunking and refinement
│
├── vectordb.py
│   └── Pinecone vector index creation, storage and retrieval
│
├── requirements.txt
│   └── Project dependencies
│
├── sample pdf.pdf
│   └── Sample insurance policy document
│
└── 5 results.txt
    └── Sample question-answer results
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Data & NLP

* Document Processing
* Text Preprocessing
* Text Chunking
* Semantic Search
* Natural Language Processing (NLP)
* Information Retrieval

### AI / Machine Learning

* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Embedding-based Retrieval
* Prompt Engineering
* Context Grounding

### Frameworks & Libraries

* FastAPI
* LlamaParse
* Chonkie
* Pinecone
* Groq
* Pandas
* Python-dotenv

---

## ⚡ API

The project exposes a REST API for document question answering.

### Endpoint

```text
POST /hackrx/run
```

### Input

```json
{
  "documents": "document_url",
  "questions": [
    "What are the base covers?",
    "What is the entry age limit?"
  ]
}
```

### Output

```json
{
  "answers": [
    "The base covers include...",
    "The entry age limit is..."
  ]
}
```

The API processes the document, creates the vector index, retrieves relevant information for each question, and generates the final answers.

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd bajaj0909
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file:

```text
LLAMA_CLOUD_API_KEY=your_key
PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key
```

### 4. Start the API

```bash
python main.py
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## 📊 Sample Results

The system was tested on an insurance policy document with questions covering:

* Eligibility
* Entry age
* Base coverage
* Emergency hospitalization
* Personal accident coverage
* OPD treatment
* Optional covers
* Baggage loss
* Travel delay
* Claim documentation
* Claim notification timelines

The generated answers demonstrate the ability of the pipeline to retrieve relevant policy information and transform it into concise natural-language responses.

---

## 💡 Key Technical Concepts Demonstrated

This project demonstrates practical implementation of:

**Document AI → NLP → Chunking → Semantic Retrieval → Vector Database → RAG → LLM Generation → REST API**

More specifically:

* Designing a document-processing pipeline
* Handling unstructured PDF data
* Chunking long documents for retrieval
* Building a vector-search workflow
* Semantic information retrieval
* Connecting retrieved context with an LLM
* Designing prompts for grounded generation
* Building an API around an AI pipeline
* Measuring and logging processing time

---

## 🔮 Future Improvements

Potential improvements include:

* Retrieval evaluation using Precision@K / Recall@K
* RAGAS-based RAG evaluation
* Hybrid keyword + semantic retrieval
* Reranking retrieved chunks
* Metadata-based filtering
* Query expansion
* Conversation memory
* Support for multiple documents
* Citation-level source attribution
* Caching embeddings and indexes
* Batch question processing optimization
* Automated evaluation dataset for measuring answer accuracy

---

## ⚠️ Security

API credentials should **never be committed to GitHub**.

Store credentials in environment variables using `.env` and add the file to `.gitignore`.

---

## 📌 Project Summary

**NeuroGuard** explores multimodal physiological data analysis, while this project focuses on **NLP, information retrieval, vector databases, and LLM-based question answering**.

Together, these projects demonstrate practical exposure to multiple areas of modern Data Science and AI:

```text
Data Processing
      +
Feature / Information Extraction
      +
Machine Learning / AI
      +
NLP
      +
Information Retrieval
      +
LLMs
      +
Data-driven Applications
```

This project was built to understand and implement an **end-to-end RAG pipeline for real-world unstructured documents**, rather than simply calling an LLM API.
