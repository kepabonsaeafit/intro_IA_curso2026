# ResearchCrew

Sistema multi-agente para debate académico sobre literatura científica en IA.
Sube papers en PDF sobre **LLMs vs ML tradicional**, haz una pregunta de investigación
y obtén un debate estructurado entre agentes con personalidades distintas.

**Curso:** Introducción a la Inteligencia Artificial — EAFIT 2026-1

---

## Arquitectura

```
PDFs + pregunta
      ↓
Search Agent (DDGS) → Reader Agent (RAG + FAISS)
                              ↓
             ┌────────────────┴────────────────┐
        Hype Agent                    Contrarian Agent
        (Mistral, pro-LLMs)          (GLM, pro-ML clásico)
             └────────────────┬────────────────┘
                         Árbitro (Llama)
                              ↓
                    Writer Agent + Pyvis
                              ↓
              Reporte Markdown + Mapa de conceptos
```

El flujo corre sobre un `StateGraph` de LangGraph. Hype y Contrarian se ejecutan en paralelo.

---

## Requisitos

- Python 3.11+
- Cuenta en [NVIDIA NIM](https://build.nvidia.com/) con créditos disponibles

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
copy .env.example .env       # Windows
# cp .env.example .env       # Linux / Mac
```

Edita `.env` y agrega tu `NVIDIA_API_KEY`:

```env
NVIDIA_API_KEY=nvapi-...
```

---

## Ejecución

```bash
uvicorn app.main:app --reload
```

Abre [http://localhost:8000](http://localhost:8000) en el navegador.

---

## Uso

1. Escribe una pregunta de investigación sobre LLMs vs ML tradicional
2. Sube uno o más papers en PDF
3. Haz clic en **Iniciar debate**
4. Espera ~3-5 minutos mientras los agentes trabajan
5. Lee el debate, la síntesis del árbitro y el reporte final

Los reportes se guardan en `app/outputs/reports/` y los grafos en `app/outputs/graphs/`.

---

## Tests

```bash
pytest tests/ -v
```

Requiere `NVIDIA_API_KEY` configurada. El test de `reader_agent` requiere al menos un PDF en `papers/`.

---

## Modelos

| Agente | Modelo (producción) | Variable |
|---|---|---|
| Search + Reader | gemma-3-4b-it (dev) | `DEV_MODEL` |
| El Hype | mistralai/mistral-large | `HYPE_MODEL` |
| El Contrarian | z-ai/glm-5.1 | `CONTRARIAN_MODEL` |
| Árbitro + Writer | meta/llama-3.3-70b-instruct | `ARBITRO_MODEL` / `WRITER_MODEL` |

Durante desarrollo todos los agentes usan `DEV_MODEL` para ahorrar créditos.
Cambia los modelos en `.env` cuando el flujo completo funcione.
