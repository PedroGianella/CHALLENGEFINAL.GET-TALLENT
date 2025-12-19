import os
from dotenv import load_dotenv

import cohere
import chromadb
from chromadb.config import Settings

# Configuración
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("Falta COHERE_API_KEY en el .env")

EMBEDDING_MODEL = "embed-multilingual-v2.0"

CHROMA_PATH = "./chroma_rag_api"
COLLECTION_NAME = "talleres_rag"

co = cohere.Client(api_key=COHERE_API_KEY)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)


# Chunking 
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 200):
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks



# Lectura del TXT
TXT_FILE = "talleres.txt"
with open(TXT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

print(f"[INFO] Texto cargado: {len(full_text)} caracteres")
if len(full_text) < 100_000:
    print("[WARN] El documento tiene menos de 100.000 caracteres (requisito mínimo).")

chunks = chunk_text(full_text, chunk_size=900, overlap=200)
print(f"[INFO] Total de chunks: {len(chunks)}")


# Reiniciar colección
print("[INFO] Reiniciando colección en Chroma...")
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# Embeddings
print("[INFO] Generando embeddings con Cohere...")

embed_resp = co.embed(
    texts=chunks,
    model=EMBEDDING_MODEL,
    input_type="search_document"
)

embeddings = embed_resp.embeddings
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"source": TXT_FILE, "chunk_index": i} for i in range(len(chunks))]

# Guardado por partes
BATCH = 96
for i in range(0, len(chunks), BATCH):
    j = i + BATCH
    collection.add(
        ids=ids[i:j],
        documents=chunks[i:j],
        embeddings=embeddings[i:j],
        metadatas=metadatas[i:j]
    )
    print(f"[INFO] Guardados {min(j, len(chunks))} / {len(chunks)} chunks")

print(f"[SUCCESS] Base vectorial lista (persistente) en: {CHROMA_PATH}")

