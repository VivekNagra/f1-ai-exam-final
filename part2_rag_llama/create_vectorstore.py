import pandas as pd
import warnings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


base_path = "../data/f1Data/"
results = pd.read_csv(base_path + "results.csv")
drivers = pd.read_csv(base_path + "drivers.csv")
constructors = pd.read_csv(base_path + "constructors.csv")
races = pd.read_csv(base_path + "races.csv")
circuits = pd.read_csv(base_path + "circuits.csv")
status = pd.read_csv(base_path + "status.csv")


df = results \
    .merge(drivers, on="driverId", how="left") \
    .merge(constructors, on="constructorId", how="left", suffixes=("", "_constructor")) \
    .merge(races, on="raceId", how="left", suffixes=("", "_race")) \
    .merge(circuits, on="circuitId", how="left", suffixes=("", "_circuit")) \
    .merge(status, on="statusId", how="left")


df = df.rename(columns={
    "forename": "driver_forename",
    "surname": "driver_surname",
    "name": "constructor_name",        
    "name_race": "race_name",          
    "circuitRef": "circuit_name"      
})


required_columns = ["driver_forename", "driver_surname", "constructor_name", "grid", "positionOrder", "year", "status"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("Missing columns in DataFrame:", missing)
    exit(1)


df = df.dropna(subset=required_columns)


documents = []
for _, row in df.iterrows():
    driver_name = f"{row['driver_forename']} {row['driver_surname']}"
    constructor = row['constructor_name']
    race = row.get("race_name", "Unknown Race")
    year = int(row['year'])
    grid = int(row['grid'])
    position = int(row['positionOrder'])
    status_text = row['status']
    circuit = row.get("circuit_name", "Unknown Circuit")

    content = (
        f"In the {year} {race} at {circuit}, {driver_name} ({constructor}) "
        f"started from grid position {grid} and finished at position {position}. "
        f"Status: {status_text}."
    )
    documents.append(Document(page_content=content))


splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = splitter.split_documents(documents)

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_function,
    persist_directory="./chromadb"
)

vectorstore.persist()
print("Vector store created with readable names and saved to ./chromadb")
