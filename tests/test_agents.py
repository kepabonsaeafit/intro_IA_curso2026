"""
Tests individuales por agente — todos usan DEV_MODEL para no gastar créditos.
Requiere NVIDIA_API_KEY en el .env o en el entorno.

Correr: pytest tests/ -v
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

# Saltar todos los tests si no hay API key
pytestmark = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY no configurada — agrega tu clave en .env",
)

SAMPLE_CONTEXT = """
Los LLMs (Large Language Models) como GPT-4 han demostrado capacidades emergentes en
razonamiento, traducción y generación de código. Se basan en arquitecturas transformer
con miles de millones de parámetros entrenados sobre texto masivo.

El ML tradicional — XGBoost, random forests, SVMs — domina en datos tabulares,
ofrece mayor interpretabilidad y requiere órdenes de magnitud menos cómputo.
Estudios recientes muestran que XGBoost supera a los LLMs en benchmarks tabulares.
"""

HYPE_ARGUMENT = (
    "Los LLMs representan un salto cualitativo: sus capacidades emergentes, "
    "in-context learning y versatilidad los hacen superiores para tareas complejas."
)

CONTRARIAN_ARGUMENT = (
    "El ML clásico es más eficiente, interpretable y sigue ganando en datos tabulares. "
    "Los LLMs son costosos y opacos para la mayoría de casos de negocio reales."
)


def test_search_agent():
    """search_agent debe retornar contexto web no vacío."""
    from app.agents.search_agent import search_agent

    result = search_agent(
        "¿Superan los LLMs al ML clásico en datos tabulares?",
        verbose=False,
    )

    assert isinstance(result, str)
    assert len(result) > 100, "El search_agent retornó muy poco contenido"


def test_hype_agent():
    """hype_agent debe retornar un argumento defendiendo los LLMs."""
    from app.agents.hype_agent import hype_agent

    result = hype_agent(
        question="¿Son los LLMs superiores al ML tradicional?",
        shared_context=SAMPLE_CONTEXT,
        verbose=False,
    )

    assert isinstance(result, str)
    assert len(result) > 50, "El hype_agent retornó una respuesta demasiado corta"


def test_contrarian_agent():
    """contrarian_agent debe retornar un argumento defendiendo el ML clásico."""
    from app.agents.contrarian_agent import contrarian_agent

    result = contrarian_agent(
        question="¿Son los LLMs superiores al ML tradicional?",
        shared_context=SAMPLE_CONTEXT,
        verbose=False,
    )

    assert isinstance(result, str)
    assert len(result) > 50, "El contrarian_agent retornó una respuesta demasiado corta"


def test_arbitro_agent():
    """arbitro_agent debe sintetizar los dos argumentos en consenso neutral."""
    from app.agents.arbitro_agent import arbitro_agent

    result = arbitro_agent(
        question="¿Cuál enfoque es mejor para aplicaciones reales de IA?",
        hype_argument=HYPE_ARGUMENT,
        contrarian_argument=CONTRARIAN_ARGUMENT,
        verbose=False,
    )

    assert isinstance(result, str)
    assert len(result) > 50, "El arbitro_agent retornó una respuesta demasiado corta"


def test_writer_agent():
    """writer_agent debe retornar reporte Markdown y HTML del grafo Pyvis."""
    from app.agents.writer_agent import writer_agent

    result = writer_agent(
        question="¿LLMs o ML clásico para aplicaciones empresariales?",
        synthesis="Ambos enfoques tienen ventajas según el contexto y los datos disponibles.",
        shared_context=SAMPLE_CONTEXT,
        verbose=False,
    )

    assert isinstance(result, dict), "writer_agent debe retornar un dict"
    assert "report"    in result,    "Falta la clave 'report'"
    assert "graph_html" in result,   "Falta la clave 'graph_html'"
    assert len(result["report"]) > 100,    "El reporte es demasiado corto"
    assert "<html" in result["graph_html"].lower(), "graph_html no parece HTML válido"

    # Verificar que los archivos se guardaron a disco
    assert Path(result["report_path"]).exists(), "El archivo de reporte no existe en disco"
    assert Path(result["graph_path"]).exists(),  "El archivo de grafo no existe en disco"


@pytest.mark.skipif(
    not list(Path("papers").glob("*.pdf")),
    reason="No hay PDFs en papers/ — agrega al menos un PDF para probar el reader_agent",
)
def test_reader_agent():
    """reader_agent debe extraer contexto relevante de los PDFs subidos."""
    from app.agents.reader_agent import reader_agent

    pdf_paths = [str(p) for p in Path("papers").glob("*.pdf")][:1]

    result = reader_agent(
        question="¿Cuál es la contribución principal de este paper?",
        pdf_paths=pdf_paths,
        verbose=False,
    )

    assert isinstance(result, str)
    assert len(result) > 50, "El reader_agent retornó muy poco contenido"
