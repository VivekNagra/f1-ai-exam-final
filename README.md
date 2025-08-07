# 🏎️ F1 AI Exam Project – Formula 1 Insights with ML & RAG

This repository contains the final project for the AI course exam. It includes:

- **Part 1:** A supervised ML Streamlit dashboard analyzing F1 race results.
- (dataset can be downloaded here: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **Part 2:** A Retrieval-Augmented Generation (RAG) system using local LLaMA2 via Ollama.

---

## Project Structure

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/VivekNagra/f1-ai-exam-final.git
cd f1-ai-exam-final
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure you **also** have:
- [Ollama installed](https://ollama.com/) and running locally with `llama2` pulled.

```bash
ollama run llama2
```

---

## Part 1: Streamlit ML Dashboard

Inside the `part1_presentation/` folder.

### To run the app:

```bash
cd part1_presentation
streamlit run part1_presentation.py
```

> This loads a dashboard where users can explore predictions like:  
> "Will a driver finish the race?" using classification models.

---

## Exam Part 2: RAG-based CLI AI with Local LLaMA2

### 1. Build the Vector Store

This step loads and enriches the CSV files with driver and constructor names and prepares a vector store for querying:

```bash
cd part2_rag_llama
python create_vectorstore.py
```

You'll see:

```
Vector store created with readable names and saved to ./chromadb
```

### 2. Ask questions using CLI

```bash
python cli.py "Who had the best grid position in 2021?"
```

Sample output:

```
Answer: In the 2021 Monaco Grand Prix, Charles Leclerc (Ferrari) started from position 1 and finished at position 20. Status: Did not start.
```

---

## Dependencies

Additions to standard libraries:

```
langchain
chromadb
sentence-transformers
langchain-community
langchain-huggingface
langchain-ollama
pandas
torch
scikit-learn
streamlit
```

You can install them manually if needed:

```bash
pip install langchain chromadb sentence-transformers langchain-community langchain-huggingface langchain-ollama
```

---

## Notes

- The RAG system uses local LLaMA2 via Ollama.
- Questions can be any F1-related queries, especially around races, drivers, constructors, or positions.
- Document formatting ensures readable output — not just raw IDs.

---

## Author

**Vivek Singh Nagra**  
Copenhagen Business Academy – Software Development  
AI Final Exam/Boss, August 2025

---
Dataset: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020 
