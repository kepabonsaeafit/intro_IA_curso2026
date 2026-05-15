import json
import os
import time

from dotenv import load_dotenv
from openai import InternalServerError, OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
)

MODEL = os.getenv("DEV_MODEL",) # Cambiar en tu .env por el modelo que quieras usar


def run_agent(
    user_message: str,
    system: str = None,
    tools: list = None,
    tool_registry: dict = None,
    verbose: bool = True,
    label: str = "Agente",
    model: str = None,
) -> str:

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    for iteration in range(1, 11):
        if verbose:
            print(f"[{label}] iteración {iteration}")

        call_kwargs = {"model": model or MODEL, "messages": messages}
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                response = client.chat.completions.create(**call_kwargs)
                break
            except InternalServerError as e:
                if "504" in str(e) and attempt < 2:
                    print(f"[{label}] 504 timeout, reintentando en 10s... ({attempt + 1}/3)")
                    time.sleep(10)
                else:
                    raise

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "stop":
            text = choice.message.content or ""
            if verbose:
                print(f"[{label}] respuesta final:\n{text}\n")
            return text

        if finish_reason == "tool_calls":
            assistant_msg = choice.message
            messages.append(assistant_msg)

            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"[{label}] llamando tool: {tool_name}({tool_args})")

                fn = tool_registry.get(tool_name) if tool_registry else None
                result = fn(**tool_args) if fn else f"Error: herramienta '{tool_name}' no encontrada."

                if verbose:
                    print(f"[{label}] resultado tool: {str(result)[:200]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
        else:
            break

    return "[Agente detenido: se alcanzó el límite de iteraciones]"
