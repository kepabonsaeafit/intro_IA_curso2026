import json
import os
from datetime import datetime
from pathlib import Path

from pyvis.network import Network

from app.agents.base import run_agent

REPORTS_DIR = Path(__file__).parent.parent / "outputs" / "reports"
GRAPHS_DIR  = Path(__file__).parent.parent / "outputs" / "graphs"

WRITER_MODEL = os.getenv("WRITER_MODEL", "minimax-m2.7")  # Cambiar en tu .env por el modelo que quieras usar


def _generate_report(
    question: str,
    synthesis: str,
    shared_context: str,
    max_iterations: int = 3,
    verbose: bool = True,
) -> str:
    """Patrón LOOP del notebook: genera → crítico evalúa → refina si no hay APPROVED."""

    # Truncar contexto para evitar timeouts — la síntesis ya captura lo esencial
    context_snippet = shared_context[:1500] + "..." if len(shared_context) > 1500 else shared_context

    # Primera versión
    report = run_agent(
        user_message=(
            f"Pregunta de investigación: {question}\n\n"
            f"Síntesis del debate:\n{synthesis}\n\n"
            f"Contexto adicional:\n{context_snippet}"
        ),
        system=(
            "Responde SIEMPRE en español. "
            "Eres un escritor académico. Genera un reporte estructurado en Markdown "
            "sobre el debate 'LLMs vs ML tradicional'. Estructura obligatoria:\n"
            "# Pregunta de investigación\n"
            "## Posición Pro-LLMs (El Hype)\n"
            "## Posición Pro-ML Clásico (El Contrarian)\n"
            "## Síntesis y consenso\n"
            "## Conclusión\n"
            "Tono académico, 400-600 palabras. Solo el reporte, sin explicaciones extra."
        ),
        model=WRITER_MODEL,
        verbose=verbose,
        label="Writer",
    )

    for _ in range(max_iterations):
        critique = run_agent(
            user_message=f"Evalúa este reporte académico:\n\n{report}",
            system=(
                "Eres un crítico académico riguroso. Evalúa el reporte en: "
                "estructura, balance entre posiciones, claridad del consenso y rigor académico.\n"
                "- Si cumple todos los criterios, responde EXACTAMENTE: APPROVED\n"
                "- Si no, da 2-3 sugerencias específicas y accionables."
            ),
            model=WRITER_MODEL,
            verbose=verbose,
            label="Critic",
        )

        # Condición de salida verificada en Python — igual que en el notebook del profe
        if "APPROVED" in critique.upper():
            break

        report = run_agent(
            user_message=(
                f"Reporte actual:\n{report}\n\n"
                f"Feedback del crítico:\n{critique}\n\n"
                "Reescribe el reporte incorporando todo el feedback."
            ),
            system=(
                "Eres un escritor académico que refina reportes. "
                "Incorpora TODO el feedback manteniendo la estructura Markdown. "
                "Solo devuelve el reporte mejorado, sin explicaciones."
            ),
            model=WRITER_MODEL,
            verbose=verbose,
            label="Refiner",
        )

    # Limpiar bloque ```markdown que algunos modelos agregan
    if report.strip().startswith("```"):
        lines = report.strip().split("\n")
        report = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    return report


def _build_concept_graph(synthesis: str, verbose: bool = True) -> str:
    """Extrae conceptos clave con el LLM y construye el grafo Pyvis."""

    concepts_raw = run_agent(
        user_message=(
            f"Dado este texto:\n\n{synthesis}\n\n"
            "Extrae los conceptos clave y sus relaciones. "
            "Responde SOLO con JSON válido, sin markdown ni explicaciones:\n"
            '{"nodes": ["concepto1", "concepto2"], '
            '"edges": [["concepto1", "concepto2", "relación"]]}'
            "\nMáximo 8 nodos y 10 edges."
        ),
        system=(
            "Responde SIEMPRE en español. "
            "Extractor de conceptos. Devuelve únicamente JSON válido. "
            "Sin texto adicional, sin bloques de código markdown."
        ),
        model=WRITER_MODEL,
        verbose=verbose,
        label="ConceptExtractor",
    )

    try:
        clean = concepts_raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except (json.JSONDecodeError, IndexError, ValueError):
        # Fallback con conceptos del dominio si el LLM no devuelve JSON válido
        data = {
            "nodes": ["LLMs", "ML Clásico", "Transformers", "XGBoost",
                      "Interpretabilidad", "Escalabilidad"],
            "edges": [
                ["LLMs", "Transformers", "basado en"],
                ["ML Clásico", "XGBoost", "incluye"],
                ["ML Clásico", "Interpretabilidad", "ventaja"],
                ["LLMs", "Escalabilidad", "ventaja"],
                ["LLMs", "ML Clásico", "debate"],
            ],
        }

    net = Network(height="500px", width="100%", bgcolor="#1a1a2e",
                  font_color="white", directed=False, cdn_resources="remote")

    added_nodes = set()
    for node in data.get("nodes", []):
        net.add_node(node, label=node, color="#4cc9f0")
        added_nodes.add(node)

    for edge in data.get("edges", []):
        if len(edge) >= 2:
            src, dst = edge[0], edge[1]
            # Auto-agregar endpoints que el LLM olvidó listar en "nodes"
            for endpoint in (src, dst):
                if endpoint not in added_nodes:
                    net.add_node(endpoint, label=endpoint, color="#4cc9f0")
                    added_nodes.add(endpoint)
            label = edge[2] if len(edge) > 2 else ""
            net.add_edge(src, dst, title=label, color="#f72585")

    net.set_options('{"physics": {"stabilization": {"iterations": 100}}}')
    return net.generate_html()


def writer_agent(
    question: str,
    synthesis: str,
    shared_context: str,
    verbose: bool = True,
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report    = _generate_report(question, synthesis, shared_context, verbose=verbose)
    graph_html = _build_concept_graph(synthesis, verbose=verbose)

    report_path = REPORTS_DIR / f"report_{timestamp}.md"
    graph_path  = GRAPHS_DIR  / f"graph_{timestamp}.html"

    report_path.write_text(report, encoding="utf-8")
    graph_path.write_text(graph_html, encoding="utf-8")

    if verbose:
        print(f"[Writer] Reporte → {report_path}")
        print(f"[Writer] Grafo   → {graph_path}")

    return {
        "report":      report,
        "graph_html":  graph_html,
        "report_path": str(report_path),
        "graph_path":  str(graph_path),
    }
