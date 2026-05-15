import os

from app.agents.base import run_agent


def contrarian_agent(question: str, shared_context: str, verbose: bool = True) -> str:
    return run_agent(
        user_message=(
            f"Pregunta de investigación: {question}\n\n"
            f"Contexto recopilado:\n{shared_context}\n\n"
            "Construye tu argumento defendiendo el ML tradicional basándote en este contexto."
        ),
        system=(
            "Eres El Contrarian, un investigador escéptico y pragmático que defiende el ML clásico. "
            "Tu posición es que los modelos tradicionales — árboles de decisión, XGBoost, "
            "random forests, SVMs — siguen siendo superiores en la mayoría de casos reales. "
            "Argumenta con rigor sobre: interpretabilidad, eficiencia computacional, rendimiento "
            "en datos tabulares, necesidad de datos masivos de los LLMs y falta de garantías. "
            "Cuestiona las afirmaciones exageradas sobre LLMs. Usa evidencia del contexto proporcionado. "
            "Responde en 3-4 párrafos sólidos."
        ),
        model=os.getenv("CONTRARIAN_MODEL", "minimax-m2.7"), # Cambiar en tu .env por el modelo que quieras usar
        verbose=verbose,
        label="ElContrarian",
    )
