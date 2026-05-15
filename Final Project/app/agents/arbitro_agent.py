import os

from app.agents.base import run_agent


def arbitro_agent(
    question: str,
    hype_argument: str,
    contrarian_argument: str,
    verbose: bool = True,
) -> str:
    return run_agent(
        user_message=(
            f"Pregunta de investigación: {question}\n\n"
            f"--- Argumento de El Hype (pro-LLMs) ---\n{hype_argument}\n\n"
            f"--- Argumento de El Contrarian (pro-ML clásico) ---\n{contrarian_argument}\n\n"
            "Sintetiza ambas posiciones en un consenso académico neutral."
        ),
        system=(
            "Eres El Árbitro, un investigador senior imparcial y riguroso. "
            "Has escuchado dos posiciones opuestas sobre LLMs vs ML tradicional. "
            "Tu tarea es sintetizar el debate identificando:\n"
            "- Los puntos válidos de cada posición\n"
            "- Las áreas de acuerdo genuino\n"
            "- Las tensiones irresolubles y por qué existen\n"
            "- Una conclusión matizada que refleje el estado real del campo\n"
            "No tomes partido. Sé justo con ambos argumentos. "
            "Responde en 3-4 párrafos con tono académico neutral."
        ),
        model=os.getenv("ARBITRO_MODEL", "mistral-large-3-675b-instruct-2512"),# Cambiar en tu .env por el modelo que quieras usar
        verbose=verbose,
        label="ElArbitro",
    )
