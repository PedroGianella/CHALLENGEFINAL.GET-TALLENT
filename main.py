import os
import re
from typing import Optional, Tuple, List

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import cohere
import chromadb
from chromadb.config import Settings


#Configuración
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("Falta COHERE_API_KEY en el .env")

#Embeddings
EMBEDDING_MODEL = "embed-multilingual-v2.0"

CHAT_MODEL = "command-r-08-2024"

CHROMA_PATH = "./chroma_rag_api"
COLLECTION_NAME = "talleres_rag"

TOP_K = 8
SIMILARITY_THRESHOLD = 0.18
MAX_CONTEXT_CHARS = 4000

BANNED_WORDS = ["insulto", "odio", "racista"]

ANSWER_CACHE = {}

co = cohere.Client(api_key=COHERE_API_KEY)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

app = FastAPI(title="RAG Talleres - Challenge Final")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#Schemas
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    context_used: Optional[str]
    similarity_score: Optional[float]
    grounded: bool


#Utilidades
def is_inappropriate(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in BANNED_WORDS)

def normalize_question(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

def embed_query(query: str) -> List[float]:
    resp = co.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="search_query"
    )
    return resp.embeddings[0]

def cosine_distance_to_similarity(dist: float) -> float:
    """
    En Chroma con cosine: la distancia suele estar entre 0 y 2 (dependiendo implementación).
    Convertimos a similitud de forma segura en [0, 1]:
      sim = max(0, 1 - dist)
    """
    return max(0.0, 1.0 - float(dist))

def retrieve_context(query: str, top_k: int = TOP_K) -> Tuple[str, Optional[float]]:
    q_emb = embed_query(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs or not dists:
        return "", None

    sims = [cosine_distance_to_similarity(d) for d in dists]
    best_sim = max(sims) if sims else None


    context = "\n\n---\n\n".join(docs)
    return context, best_sim
#PROMPT
def build_prompt(context: str, question: str) -> str:
    return f"""
Sos un asistente especializado EXCLUSIVAMENTE en información del Club Atlético Talleres de Córdoba.
Respondés SOLO con el CONTEXTO provisto.

REGLAS:
1) Respondé SIEMPRE en español.
2) NO uses emojis.
3) Respuesta corta: máximo 3 oraciones.
4) NO inventes información.
5) Si la respuesta no está en el contexto, respondé EXACTAMENTE:
   "No cuento con información suficiente para responder a esta consulta."

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
""".strip()

def cohere_chat_generate(prompt: str) -> str:
    """
    Compatibilidad con SDK viejo y nuevo.
    - SDK viejo: co.chat(message=..., preamble=...)
    - SDK nuevo: co.chat(messages=[...])
    """
    
    try:
        resp = co.chat(
            model=CHAT_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.message.content[0].text.strip()
    except TypeError:
      
        resp = co.chat(
            model=CHAT_MODEL,
            temperature=0,
            message=prompt,
        )
       
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
        
        return str(resp).strip()


#Endpoint
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = normalize_question(req.question)

    # Cache determinista
    if question in ANSWER_CACHE:
        return ANSWER_CACHE[question]

    #Filtro lenguaje
    if is_inappropriate(question):
        resp = AskResponse(
            answer="No puedo responder a este tipo de consultas.",
            context_used=None,
            similarity_score=None,
            grounded=False
        )
        ANSWER_CACHE[question] = resp
        return resp

    #Recuperación
    try:
        context, best_sim = retrieve_context(question, top_k=TOP_K)
    except Exception as e:
        print("[ERROR retrieve]", e)
        resp = AskResponse(
            answer="El servicio externo no pudo procesar la solicitud en este momento.",
            context_used=None,
            similarity_score=None,
            grounded=False
        )
        ANSWER_CACHE[question] = resp
        return resp

    #Grounding: si no hay contexto o similitud baja
    if (not context) or (best_sim is None) or (best_sim < SIMILARITY_THRESHOLD):
        resp = AskResponse(
            answer="No cuento con información suficiente para responder a esta consulta.",
            context_used=(context[:300] if context else None),
            similarity_score=best_sim,
            grounded=False
        )
        ANSWER_CACHE[question] = resp
        return resp

    #Generación (corto y controlado)
    context = context[:MAX_CONTEXT_CHARS]
    prompt = build_prompt(context, question)

    try:
        answer = cohere_chat_generate(prompt)
    except Exception as e:
        print("[ERROR chat]", e)
        resp = AskResponse(
            answer="El servicio externo no pudo procesar la solicitud en este momento.",
            context_used=context[:300],
            similarity_score=best_sim,
            grounded=False
        )
        ANSWER_CACHE[question] = resp
        return resp

    resp = AskResponse(
        answer=answer,
        context_used=context[:300],
        similarity_score=best_sim,
        grounded=True
    )
    ANSWER_CACHE[question] = resp
    return resp

