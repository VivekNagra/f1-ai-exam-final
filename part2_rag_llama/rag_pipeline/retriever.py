from langchain_core.documents import Document
import pandas as pd

class RetrieverTool:
    def load_documents_from_csv(self, csv_path: str) -> list[Document]:
        df = pd.read_csv(csv_path)

        # Optional: Drop rows with missing essential values
        df = df.dropna(subset=['raceId', 'driverId', 'constructorId', 'positionOrder'])

        documents = []
        for _, row in df.iterrows():
            content = (
                f"Race ID: {row['raceId']}, "
                f"Driver ID: {row['driverId']}, "
                f"Constructor ID: {row['constructorId']}, "
                f"Position: {row['positionOrder']}, "
                f"Grid: {row['grid']}, "
                f"Laps: {row['laps']}, "
                f"Status ID: {row['statusId']}"
            )
            documents.append(Document(page_content=content))

        return documents
