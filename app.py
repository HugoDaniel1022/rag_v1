import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# ---------------------------
# Cargar API key
# ---------------------------
load_dotenv()
#api_key = os.getenv("GENAI_API_KEY")
api_key = st.secrets["GENAI_API_KEY"]
# ---------------------------
# Recursos cacheados
# ---------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
    return Chroma(persist_directory="db_psico", embedding_function=embeddings)

@st.cache_resource
def load_client():
    return genai.Client(api_key=api_key)

vectorstore = load_vectorstore()
client = load_client()

# ---------------------------
# Interfaz de Streamlit
# ---------------------------
st.title("Psicosaberes")
st.write("Haz todas las preguntas de lo que quieras saber sobre la Psicología")

query = st.text_input("Ingresa tu pregunta:")

if st.button("Buscar respuesta"):
    if query.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        # ---------------------------
        # Recuperación de documentos
        # ---------------------------
        docs = vectorstore.similarity_search(query, k=3)

        # Crear contexto combinando los chunks
        context = "\n".join([d.page_content[:1000] for d in docs])  # limitar tamaño por chunk

        # ---------------------------
        # Prompt para Gemini
        # ---------------------------
        prompt = f"""
Usa el siguiente contexto para responder la pregunta de manera clara y completa.

CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""

        # ---------------------------
        # Generar respuesta con Gemini
        # ---------------------------
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = response.text
        except Exception as e:
            answer = f"⚠️ Error al generar la respuesta: {e}"

        # ---------------------------
        # Mostrar resultado
        # ---------------------------
        st.subheader("💬 Respuesta")
        st.write(answer)

        # Mostrar contexto completo usado solo si se quiere
        with st.expander("📄 Contexto completo usado: "):
            st.text(context)
