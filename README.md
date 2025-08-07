# Formula 1 AI Exam Project

This repository contains both **Part 1 (Supervised Learning Dashboard)** and **Part 2 (RAG-based Question Answering with LLaMA2)** for the AI exam project.


## PART 1: Supervised Learning Dashboard (Streamlit)

### Step 1: Install dependencies
Use  venv.
with venv:

```bash
python3 -m venv venv
source venv/bin/activate
cd part1_presentation
pip install -r requirements.txt
```

### Step 2: Run the Jupyter Notebook

Run the analysis notbook (run all cells) and train the model 
to start the notebook run the command (from inside the dir part1_presentation )

```bash
jupyter notebook 
```

This will generate a trained `model.pkl`, inside the part1_presentation fodler.


### Step 3: Run the Streamlit Dashboard
exit the notebook server if not alr done (ctrl+c)

the run the command:

```bash
streamlit run part1_presentation.py
```

from here you should see the streamlit app with the part 1 app:
<img width="1373" height="1014" alt="image" src="https://github.com/user-attachments/assets/abf66bd1-1f94-4f09-8e9b-6117e9dd06c6" />


---

## PART 2: RAG-based AI with LLaMA2 and ChromaDB

cd to part 2 "cd part2_rag_llama"

### Step 1: Set up a virtual environment
installing the requirements for this venv2 will take a while..

```bash
python3.11 -m venv venv2
source venv2/bin/activate
pip install -r requirements.txt
```

Make sure Ollama is running with LLaMA2 and OpenWebUI (Docker-based).
start dokcer
and run this:
```bash
docker run -d -p 3000:3000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --name openwebui \
  ghcr.io/open-webui/open-webui:main
```

run this in a terminal:
```bash
ollama run llama2 (keep this terminal running)
```

### Step 2: Create the vectorstore

```bash
cd part2_rag_llama
python create_vectorstore.py
```

This will embed and persist the documents into `./chromadb/`.

### Step 3: Ask a Question via CLI

```bash
python cli.py "Who had the best grid position in the race?"
```

---

## Troubleshooting Part 1 (model.pkl errors)

If you get a `TypeError` related to NumPy or `joblib`:

### Option 1: Retrain the model
Re-run the notebook in your current virtual environment (remember to run requirements filde inside part1 folder) and re-save:

```python
joblib.dump(model, "../part1_presentation/model.pkl")
```

## Requirements

Install all dependencies using:
remember to use python 3.11 for part 2!!

```bash
pip install -r requirements.txt
```

Contents:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
jupyter
joblib
langchain
chromadb
sentence-transformers
openai
tqdm
```

---

## 📄 License

MIT License.
