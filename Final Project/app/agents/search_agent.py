from ddgs import DDGS

from app.agents.base import run_agent

# --- Tool implementation ---

def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No se encontraron resultados para la búsqueda."
    output = []
    for r in results:
        output.append(f"**{r['title']}**\n{r['body']}\nFuente: {r['href']}")
    return "\n---\n".join(output)


# --- Tool schema (formato OpenAI, igual que en el notebook del profe) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Busca información académica actualizada en la web. "
                "Úsala para encontrar contexto sobre papers, autores, "
                "benchmarks o tendencias recientes en IA."
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

TOOL_REGISTRY = {
    "web_search": web_search,
}

# --- Agente ---

def search_agent(question: str, verbose: bool = True) -> str:
    return run_agent(
        user_message=(
            f"Busca información académica relevante sobre esta pregunta de investigación:\n\n"
            f"{question}\n\n"
            "Haz 2-3 búsquedas para cubrir distintos ángulos del tema. "
            "Presenta un resumen con las fuentes más relevantes."
        ),
        system=(
            "Eres un agente de búsqueda académica especializado en IA. "
            "Tu tarea es recopilar contexto actualizado de la web sobre "
            "LLMs, ML tradicional y temas relacionados. "
            "Usa web_search varias veces con queries distintas para obtener perspectivas variadas. "
            "Cita siempre las fuentes."
        ),
        tools=TOOLS,
        tool_registry=TOOL_REGISTRY,
        verbose=verbose,
        label="SearchAgent",
    )
