import os
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma


# Cargar documentos PDF
docs = []

# Carpeta donde están tus PDFs
folder_path = "documentos/"

# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".pdf"):
#         loader = PyPDFLoader(os.path.join(folder_path, file_name))
#         docs += loader.load()  # cada PDF puede generar varios documentos/chunks


# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".pdf"):
#         loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
#         docs += loader.load()


for file_name in os.listdir(folder_path):
    if file_name.endswith(".pdf"):
        loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
        loaded_docs = loader.load()
        
        for d in loaded_docs:
            # Limpieza de texto
            d.page_content = (
                d.page_content.replace("\x0c", "")
                                .replace("  ", " ")
                                .strip()
            )
            d.metadata["source"] = file_name
        
        docs += loaded_docs



# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       # tamaño del fragmento
    chunk_overlap=150,     # solapamiento entre fragmentos
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)

# for idx, chunk in enumerate(chunks):
#     chunk.metadata["chunk_id"] = idx


for idx, chunk in enumerate(chunks, start=1):
    chunk.metadata["chunk_id"] = idx
    if idx % 50 == 0 or idx == len(chunks):
        print(f"Chunk {idx}/{len(chunks)} listo")

print(f"Se crearon {len(chunks)} fragmentos de texto.")

# Embeddings (usando la clase oficial)
#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1') # ++ preciso ++ lento
#embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') #modelo + preciso pero + lento
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2') # modelo rápido no muy preciso beta


# Usamos la clase Chroma para guardar los embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db_psico"  # Carpeta donde se guarda la BD
)

# Persistir la base de datos
vectorstore.persist() # en la v0.4 ya no hace falta esta línea

print("Base de conocimiento creada y guardada en 'db_psico/' ✅")