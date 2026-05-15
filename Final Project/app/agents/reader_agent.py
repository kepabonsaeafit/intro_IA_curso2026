from pathlib import Path
from typing import List

from app.agents.base import run_agent
from app.rag.chunker import load_and_chunk
from app.rag.vectorstore import build_vectorstore, get_retriever

# --- Tool schema (formato OpenAI, igual que en el notebook del profe) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Busca fragmentos relevantes en los papers académicos subidos por el usuario. "
                "Úsala para encontrar evidencia, citas o datos específicos de los documentos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La consulta de búsqueda en lenguaje natural.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# --- Agente ---

def reader_agent(question: str, pdf_paths: List[str], verbose: bool = True) -> str:
    # Construir vectorstore con los PDFs recibidos
    all_docs = []
    for path in pdf_paths:
        docs = load_and_chunk(path, title=Path(path).stem)
        all_docs.extend(docs)

    if not all_docs:
        return "No se pudo extraer contenido de los PDFs proporcionados."

    vectorstore = build_vectorstore(all_docs)
    retriever = get_retriever(vectorstore)

    # Closure — captura el retriever específico de esta llamada
    def rag_search(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No se encontraron fragmentos relevantes para esta consulta."
        results = []
        for doc in docs:
            title = doc.metadata.get("title", "Sin título")
            results.append(f"**[{title}]**\n{doc.page_content}")
        return "\n---\n".join(results)

    tool_registry = {"rag_search": rag_search}

    return run_agent(
        user_message=(
            f"Busca en los papers académicos información relevante sobre:\n\n"
            f"{question}\n\n"
            "Haz varias búsquedas con ángulos distintos para cubrir el tema. "
            "Cita el paper de origen para cada hallazgo."
        ),
        system=(
            "Eres un agente de análisis de literatura científica. "
            "Tu tarea es extraer evidencia, datos y argumentos de los papers académicos "
            "para responder la pregunta de investigación. "
            "Usa rag_search con diferentes queries para encontrar perspectivas variadas. "
            "Organiza los hallazgos por paper y cita textualmente cuando sea relevante."
        ),
        tools=TOOLS,
        tool_registry=tool_registry,
        verbose=verbose,
        label="ReaderAgent",
    )
