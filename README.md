# CONSULTOR PSICOLÓGICO ONLINE

## Descripción

Sistema de Retrieval-Augmented Generation (RAG) que permite contestar preguntas consultando varios manuales de psicología.

Desarrollado como Trabajo Integrador N°2 para la materia Procesamiento del Habla e Introducción a LLMs (IFTS 24).

## Demo

[link](https://consultorpsicologico.streamlit.app/)

## Problema que Resuelve

El problema que resuelve es poder tener una gran cantidad de información que consultar, embebida y resumida en una base de datos vectorial.

## Arquitectura del Sistema

### Pipeline RAG

1. **Ingesta**: Se utiliza PyMuPDFLoader para cargar documentos PDF desde una carpeta de forma local
2. **Chunking**: Descubrimos que funciona mejor con chunk_size=1500 y chunk_overlap=150
3. **Embeddings**: model_name='sentence-transformers/all-MiniLM-L12-v2' (+ rápido - eficiente)
4. **Almacenamiento**: ChromaDB
5. **Retrieval**: vectorstore.similarity_search(query, k=3)
6. **Generation**: model="gemini-2.5-flash"
7. **Interfaz**: Streamlit

### Diagrama de Flujo

![Imagen](mi_imagen.png)


## Stack Tecnológico

- **LLM**: Gemini
- **Embeddings**: Hugging Face
- **Vector Database**: ChromaDB
- **Orquestación**: LangChain
- **Interfaz**: Streamlit
- **Deployment**: Streamlit Cloud
- **Otras librerías**: streamlit,python-dotenv,langchain,langchain-community,chromadb,sentence-transformers,google-genai,langchain_huggingface

## Corpus de Documentos

- **Dominio**: Psicología
- **Cantidad**: 10
- **Fuente**: Libros online
- **Formato**: PDF
- **Idioma**: Español

## Instalación y Uso Local

### Prerrequisitos

- Python 3.9+
- Gemini API key correspondiente

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/HugoDaniel1022/rag_v1.git
cd rag_v1
Crear entorno virtual:
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
Instalar dependencias:
pip install -r requirements.txt
Configurar variables de entorno (si aplica):
# Crear archivo .env con:
GEMINI_API_KEY=tu_api_key
# O la configuración que necesites
[Si es primera vez] Procesar documentos:
python indexacion.py
Ejecutar la aplicación:
streamlit run app.py
Abrir en navegador: http://localhost:8501
```

Estructura del Proyecto
├── app.py                  # Aplicación Streamlit principal
├── indexacion.py     # Script de ingesta y procesamiento
├── agregar_docs.py                # Funciones auxiliares
├── requirements.txt        # Dependencias
├── README.md              # Este archivo
├── .env           # Template de variables de entorno
├── documentos/                  # Documentos fuente
│   └── [documentos pdf]
├── chroma_db/             # Base de datos vectorial (generada)
└── .gitignore            # evita subir algunos archivos a github (opcional)


Ejemplos de Consultas
Probá estas consultas de ejemplo:

- Qué es la educación?
- En base a qué dimensiones se define El Nivel de desorden del consultante en DBT? 
- Cuales son las etapas del trastorno DBT?
- Qué es El duelo?
- En qué consiste el tratamiento para abordar el duelo?

Decisiones de Diseño
¿Por qué elegí Gemini?
Más rápidez a la hora de contestar.

¿Por qué 1500 de chunksize con 150 de overlap?
Obtubimos mejores resultados

¿Por qué top-k = 3?
Obtubimos mejores resultado

Otras decisiones importantes
Usamos Streamlit Cloud por familiaridad y facilidad para deployar el proyecto

Autor
Vanesa Cabrera & Hugo D. Peña

Trabajo Integrador N°2 Materia: Procesamiento del Habla e Introducción a LLMs Institución: IFTS 24 - Tecnicatura Superior en Ciencias de Datos e IA Profesor: Matías Barreto Año: 2025

Licencia
MIT