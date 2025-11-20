import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# 👉 Ruta de tu BD ya existente
PERSIST_DIR = "db_psico"

# 👉 Ruta donde están los nuevos PDFs
NUEVOS_DOCS_DIR = "nuevos_documentos/"


def cargar_y_procesar_pdfs(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
            loaded_docs = loader.load()

            for d in loaded_docs:
                d.page_content = (
                    d.page_content.replace("\x0c", "")
                                  .replace("  ", " ")
                                  .strip()
                )
                d.metadata["source"] = file_name

            docs += loaded_docs

    if not docs:
        print("⚠️ No se encontraron PDFs nuevos.")
    return docs


def chunkear_documentos(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)



print("📄 Cargando nuevos documentos...")
docs = cargar_y_procesar_pdfs(NUEVOS_DOCS_DIR)
if not docs:
    print("No hay documentos nuevos que cargar...")

print("✂️ Dividiendo en chunks...")
chunks = chunkear_documentos(docs)

# Embeddings (mismo modelo que usaste al crear la BD)
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L12-v2'
)

print("📦 Cargando BD existente...")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)

print(f"➕ Agregando {len(chunks)} nuevos chunks a la BD...")
vectorstore.add_documents(chunks)

vectorstore.persist()
print("✅ Documentos añadidos correctamente.")
