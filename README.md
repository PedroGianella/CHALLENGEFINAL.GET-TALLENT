# Asistente RAG – Club Atlético Talleres de Córdoba

Este proyecto implementa un **asistente conversacional basado en arquitectura RAG (Retrieval Augmented Generation)** que responde preguntas utilizando exclusivamente información relacionada con el Club Atlético Talleres de Córdoba.

La idea surge como una posible **oportunidad de negocio**, ya que el club no cuenta actualmente con un chatbot informativo para socios e hinchas.

El asistente:
- Responde solo con información presente en los documentos cargados
- Evita inventar respuestas (grounding estricto)
- Rechaza preguntas fuera de contexto
- Devuelve siempre respuestas en español

---

## Arquitectura general (RAG)

El sistema está compuesto por tres partes principales:

1. **Ingesta**
   - Se carga un archivo de texto (`talleres.txt`) con información del club.
   - El texto se divide en fragmentos (chunks).
   - Se generan embeddings usando Cohere.
   - Los embeddings se almacenan en una base vectorial persistente (ChromaDB).

2. **Recuperación**
   - Ante una pregunta del usuario, se genera un embedding de la consulta.
   - Se buscan los fragmentos más similares en la base vectorial.
   - Se valida la similitud para evitar respuestas incorrectas.

3. **Generación**
   - Se construye un prompt controlado con el contexto recuperado.
   - Se genera la respuesta usando un modelo de lenguaje de Cohere.
   - Si no hay contexto suficiente, el sistema se niega a responder.

---

## 🚀 Tecnologías utilizadas

- **Python**
- **FastAPI** (API REST)
- **Cohere** (Embeddings + generación de texto)
- **ChromaDB** (Base de datos vectorial)
- **HTML / CSS / JavaScript** (Frontend simple)
- **Arquitectura RAG con grounding**

## Documentación
La documentación completa del proyecto se encuentra en el archivo:
- Documentacion_RAG_Talleres


