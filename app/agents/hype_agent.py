import os

from app.agents.base import run_agent


def hype_agent(question: str, shared_context: str, verbose: bool = True) -> str:
    return run_agent(
        user_message=(
            f"Pregunta de investigación: {question}\n\n"
            f"Contexto recopilado:\n{shared_context}\n\n"
            "Construye tu argumento defendiendo los LLMs basándote en este contexto."
        ),
        system=(
            "Eres El Hype, un investigador apasionado y entusiasta de los Large Language Models. "
            "Tu posición es que los LLMs representan un salto cualitativo en IA que el ML "
            "tradicional no puede igualar. "
            "Argumenta con convicción sobre: arquitecturas transformer, emergent capabilities, "
            "scaling laws, razonamiento en contexto (in-context learning) y versatilidad. "
            "Usa evidencia del contexto proporcionado. Sé persuasivo pero académicamente riguroso. "
            "Responde en 3-4 párrafos sólidos."
        ),
        model=os.getenv("HYPE_MODEL", "mistralai/mistral-large"),
        verbose=verbose,
        label="ElHype",
    )
