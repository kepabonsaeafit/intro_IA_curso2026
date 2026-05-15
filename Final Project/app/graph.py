from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.arbitro_agent import arbitro_agent
from app.agents.contrarian_agent import contrarian_agent
from app.agents.hype_agent import hype_agent
from app.agents.reader_agent import reader_agent
from app.agents.search_agent import search_agent
from app.agents.writer_agent import writer_agent


# --- Estado compartido entre todos los nodos ---

class ResearchState(TypedDict):
    question:          str
    pdf_paths:         List[str]
    search_results:    str
    rag_results:       str
    shared_context:    str
    hype_argument:     str
    contrarian_argument: str
    synthesis:         str
    report:            str
    graph_html:        str
    report_path:       str
    graph_path:        str


# --- Nodos: un nodo por agente, firma (state) -> dict ---

def search_node(state: ResearchState) -> dict:
    results = search_agent(state["question"])
    return {"search_results": results}


def reader_node(state: ResearchState) -> dict:
    results = reader_agent(state["question"], state["pdf_paths"])
    shared_context = (
        f"=== Contexto web ===\n{state['search_results']}\n\n"
        f"=== Contexto de papers ===\n{results}"
    )
    return {"rag_results": results, "shared_context": shared_context}


def hype_node(state: ResearchState) -> dict:
    argument = hype_agent(state["question"], state["shared_context"])
    return {"hype_argument": argument}


def contrarian_node(state: ResearchState) -> dict:
    argument = contrarian_agent(state["question"], state["shared_context"])
    return {"contrarian_argument": argument}


def arbitro_node(state: ResearchState) -> dict:
    synthesis = arbitro_agent(
        state["question"],
        state["hype_argument"],
        state["contrarian_argument"],
    )
    return {"synthesis": synthesis}


def writer_node(state: ResearchState) -> dict:
    result = writer_agent(
        state["question"],
        state["synthesis"],
        state["shared_context"],
    )
    return result  # {"report", "graph_html", "report_path", "graph_path"}


# --- Construcción del grafo ---

def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("search",     search_node)
    builder.add_node("reader",     reader_node)
    builder.add_node("hype",       hype_node)
    builder.add_node("contrarian", contrarian_node)
    builder.add_node("arbitro",    arbitro_node)
    builder.add_node("writer",     writer_node)

    # Capa 1: secuencial
    builder.add_edge(START,    "search")
    builder.add_edge("search", "reader")

    # Capa 2: fan-out → hype y contrarian corren en paralelo
    builder.add_edge("reader", "hype")
    builder.add_edge("reader", "contrarian")

    # fan-in → arbitro espera a que ambos terminen
    builder.add_edge("hype",       "arbitro")
    builder.add_edge("contrarian", "arbitro")

    builder.add_edge("arbitro", "writer")
    builder.add_edge("writer",  END)

    return builder.compile()


# --- API pública ---

def run_research(question: str, pdf_paths: List[str]) -> ResearchState:
    graph = build_graph()

    initial_state: ResearchState = {
        "question":            question,
        "pdf_paths":           pdf_paths,
        "search_results":      "",
        "rag_results":         "",
        "shared_context":      "",
        "hype_argument":       "",
        "contrarian_argument": "",
        "synthesis":           "",
        "report":              "",
        "graph_html":          "",
        "report_path":         "",
        "graph_path":          "",
    }

    return graph.invoke(initial_state)
