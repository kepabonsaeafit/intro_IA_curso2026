# ResearchCrew — Guión de Sustentación
**IA EAFIT 2026-1 · Kevin Pabón**

> Este documento sirve para las dos modalidades: uno a uno con el profesor o presentación en pantalla ante audiencia. Los bloques de código son exactamente los del proyecto — no los inventes, ábrelos en el editor mientras hablas.

---

## APERTURA

"El proyecto se llama ResearchCrew. Es un sistema multi-agente que toma papers académicos en PDF y una pregunta de investigación, y produce un debate estructurado entre cuatro agentes con personalidades distintas sobre LLMs vs ML tradicional. Al final genera un reporte en Markdown y un mapa de conceptos interactivo con Pyvis."

"El stack es: FastAPI de backend, HTML/JS vanilla de frontend, LangGraph como orquestador del grafo de agentes, LangChain solo para el RAG, y NVIDIA NIM como proveedor de modelos — API compatible con OpenAI."

---

## BLOQUE 1 — ARQUITECTURA GENERAL

> Abrir: `app/graph.py`

"Todo el sistema tiene tres capas. La primera es recolección de información: el Search Agent busca contexto en la web con DuckDuckGo, y el Reader Agent hace RAG sobre los PDFs con FAISS. La segunda es el debate: El Hype y El Contrarian corren en paralelo, luego El Árbitro sintetiza. La tercera capa es el output: el Writer genera el reporte y el mapa de conceptos."

```
PDFs + pregunta
      ↓
[search_node]  → DuckDuckGo
[reader_node]  → RAG sobre PDFs
      ↓
   shared_context
      ↓
[hype_node]  ←→  [contrarian_node]   ← paralelo
      ↓                ↓
          [arbitro_node]              ← fan-in, espera ambos
               ↓
          [writer_node]               ← reporte + Pyvis
```

"El orquestador es el `StateGraph` de LangGraph definido en `build_graph()`. Cada agente es un nodo. Los edges definen el flujo."

```python
# app/graph.py — build_graph()
builder.add_edge(START,    "search")
builder.add_edge("search", "reader")

# fan-out: reader lanza hype y contrarian en paralelo
builder.add_edge("reader", "hype")
builder.add_edge("reader", "contrarian")

# fan-in: arbitro espera a que ambos terminen
builder.add_edge("hype",       "arbitro")
builder.add_edge("contrarian", "arbitro")

builder.add_edge("arbitro", "writer")
builder.add_edge("writer",  END)
```

---

## BLOQUE 2 — EL LOOP AGÉNTICO

> Abrir: `app/agents/base.py`

"El loop agéntico está en `run_agent`, en `base.py`. Es el patrón del notebook del profe adaptado a NVIDIA NIM. Corre hasta 10 iteraciones."

```python
# app/agents/base.py
def run_agent(user_message, system=None, tools=None,
              tool_registry=None, verbose=True, label="Agente", model=None):

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    for iteration in range(1, 11):
        response = client.chat.completions.create(**call_kwargs)
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            return response.choices[0].message.content   # ← LLM terminó

        if finish_reason == "tool_calls":
            # ejecuta la tool, agrega resultado al historial
            # y vuelve a llamar al LLM
```

"Si el LLM devuelve `finish_reason == 'stop'`, retorna la respuesta. Si devuelve `tool_calls`, ejecuta la herramienta Python, agrega el resultado al historial de mensajes y llama al LLM de nuevo. Eso es el loop agéntico — el modelo decide cuántas veces llama tools, no el código."

"También tiene reintentos ante errores 504 del servidor:"

```python
# app/agents/base.py — retry ante timeouts de NVIDIA NIM
for attempt in range(3):
    try:
        response = client.chat.completions.create(**call_kwargs)
        break
    except InternalServerError as e:
        if "504" in str(e) and attempt < 2:
            time.sleep(10)
        else:
            raise
```

---

## BLOQUE 3 — LAS TOOLS (Search Agent)

> Abrir: `app/agents/search_agent.py`

"Las tools siguen exactamente el formato OpenAI del notebook: un JSON schema que describe qué hace la tool, y un diccionario `TOOL_REGISTRY` que mapea nombre → función Python."

```python
# app/agents/search_agent.py
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Busca información académica actualizada en la web...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La consulta de búsqueda."}
            },
            "required": ["query"],
        },
    }
}]

TOOL_REGISTRY = {"web_search": web_search}
```

"El `run_agent` recibe ambos. El LLM ve el schema y decide cuándo llamar la tool. Python ejecuta la función real, que usa DuckDuckGo."

```python
# app/agents/search_agent.py — implementación real de la tool
def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    output = []
    for r in results:
        output.append(f"**{r['title']}**\n{r['body']}\nFuente: {r['href']}")
    return "\n---\n".join(output)
```

---

## BLOQUE 4 — RAG PIPELINE

> Abrir: `app/rag/`

"El RAG tiene tres archivos, igual que el Workshop 3 del profe."

**`app/rag/loader.py` — convierte PDF a texto:**

```python
# app/rag/loader.py
from markitdown import MarkItDown

md_converter = MarkItDown()

def load_pdf(path):
    result = md_converter.convert(str(path))
    text = result.text_content
    if not text or not text.strip():
        raise ValueError("No se pudo extraer texto. Verifica que el PDF no sea escaneado.")
    return text
```

**`app/rag/chunker.py` — divide en fragmentos con overlap:**

```python
# app/rag/chunker.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", " "],
)
```

**`app/rag/vectorstore.py` — embeddings locales y FAISS:**

```python
# app/rag/vectorstore.py
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.3},
    )
```

"Uso `all-MiniLM-L6-v2` de HuggingFace para embeddings locales — sin API key, sin rate limits. FAISS como índice local en memoria."

---

## BLOQUE 5 — READER AGENT Y EL CLOSURE

> Abrir: `app/agents/reader_agent.py`

"El Reader Agent tiene una tool `rag_search`. La particularidad es que el retriever se crea como **closure** dentro de la función del agente, porque depende de qué PDFs subió el usuario en esa sesión específica — no es global."

```python
# app/agents/reader_agent.py
def reader_agent(question: str, pdf_paths: List[str]) -> str:
    # Construye el vectorstore con los PDFs de esta sesión
    all_docs = []
    for path in pdf_paths:
        docs = load_and_chunk(path, title=Path(path).stem)
        all_docs.extend(docs)

    vectorstore = build_vectorstore(all_docs)
    retriever   = get_retriever(vectorstore)

    # Closure — captura el retriever específico de esta llamada
    def rag_search(query: str) -> str:
        docs = retriever.invoke(query)
        results = []
        for doc in docs:
            title = doc.metadata.get("title", "Sin título")
            results.append(f"**[{title}]**\n{doc.page_content}")
        return "\n---\n".join(results)

    tool_registry = {"rag_search": rag_search}

    return run_agent(..., tools=TOOLS, tool_registry=tool_registry)
```

"Si hiciera el retriever global, mezclaría los documentos de distintas sesiones. El closure lo aísla correctamente."

---

## BLOQUE 6 — LOS AGENTES DE DEBATE

> Abrir: `app/agents/hype_agent.py` y `app/agents/contrarian_agent.py`

"El Hype y El Contrarian son funciones simples que llaman a `run_agent`. No tienen tools — solo reciben el contexto compartido y argumentan desde su personalidad."

```python
# app/agents/hype_agent.py
def hype_agent(question: str, shared_context: str) -> str:
    return run_agent(
        user_message=f"Pregunta: {question}\n\nContexto:\n{shared_context}\n\n"
                     "Construye tu argumento defendiendo los LLMs.",
        system=(
            "Eres El Hype, un investigador apasionado de los LLMs. "
            "Argumenta sobre: transformers, emergent capabilities, scaling laws, "
            "in-context learning y versatilidad. Sé persuasivo pero académicamente riguroso."
        ),
        model=os.getenv("HYPE_MODEL", "mistralai/mistral-large-3-675b-instruct-2512"),
        label="ElHype",
    )
```

```python
# app/agents/contrarian_agent.py
def contrarian_agent(question: str, shared_context: str) -> str:
    return run_agent(
        system=(
            "Eres El Contrarian, un investigador escéptico que defiende el ML clásico. "
            "Argumenta sobre: interpretabilidad, eficiencia computacional, rendimiento "
            "en datos tabulares, necesidad de datos masivos de los LLMs y falta de garantías."
        ),
        model=os.getenv("CONTRARIAN_MODEL", "minimaxai/minimax-m2.7"),
        label="ElContrarian",
    )
```

"Cada uno tiene un modelo diferente: Hype usa Mistral Large 675B, Contrarian usa Minimax M2.7. Y corren en paralelo — LangGraph lanza ambos nodos cuando `reader` termina."

---

## BLOQUE 7 — PATRONES MULTI-AGENTE

> Abrir: `app/graph.py`

"Usé los cuatro patrones del notebook `part02_multiagent_architectures`, implementados como nodos del StateGraph."

| Patrón | Dónde en el código |
|---|---|
| **SEQUENTIAL** | `START → search → reader` — edges lineales, el orden importa |
| **PARALLEL** | `reader → hype` y `reader → contrarian` — fan-out simultáneo |
| **ORCHESTRATOR** | `build_graph()` completo — el StateGraph es el orquestador central |
| **LOOP** | `writer_agent.py` — Writer genera → Critic evalúa → refina si no hay APPROVED |

"El paralelismo lo maneja LangGraph automáticamente. Cuando dos edges salen del mismo nodo, LangGraph los ejecuta en paralelo. Cuando dos edges llegan al mismo nodo, espera a que ambos terminen — eso es el fan-in."

---

## BLOQUE 8 — PATRÓN LOOP (Writer)

> Abrir: `app/agents/writer_agent.py` → función `_generate_report`

"El patrón loop está en el Writer. Es el mismo del notebook: genera un borrador, un crítico lo evalúa, y si no dice APPROVED lo refina. Máximo 3 iteraciones. La condición de salida se verifica en Python, no en el LLM."

```python
# app/agents/writer_agent.py — _generate_report()
report = run_agent(...)   # primera versión del reporte

for _ in range(max_iterations):     # max 3 iteraciones
    critique = run_agent(
        system="...Si cumple todos los criterios, responde EXACTAMENTE: APPROVED...",
        label="Critic",
    )

    if "APPROVED" in critique.upper():   # condición verificada en Python
        break

    report = run_agent(
        user_message=f"Reporte actual:\n{report}\n\nFeedback:\n{critique}\n\n"
                     "Reescribe incorporando todo el feedback.",
        label="Refiner",
    )
```

"El Writer también genera el mapa de conceptos: le pide al LLM que extraiga nodos y edges en JSON, y los convierte en un grafo Pyvis interactivo."

```python
# app/agents/writer_agent.py — _build_concept_graph()
net = Network(height="500px", width="100%",
              bgcolor="#1a1a2e", font_color="white")

for node in data["nodes"]:
    net.add_node(node, label=node, color="#4cc9f0")

for edge in data["edges"]:
    net.add_edge(edge[0], edge[1], title=edge[2], color="#f72585")

return net.generate_html()
```

---

## BLOQUE 9 — ESTADO COMPARTIDO

> Abrir: `app/graph.py` → `ResearchState`

"El estado compartido es un `TypedDict` llamado `ResearchState`. Cada nodo recibe el estado completo y retorna solo los campos que modifica. LangGraph hace el merge automáticamente."

```python
# app/graph.py
class ResearchState(TypedDict):
    question:             str
    pdf_paths:            List[str]
    search_results:       str   # lo escribe search_node
    rag_results:          str   # lo escribe reader_node
    shared_context:       str   # lo construye reader_node
    hype_argument:        str   # lo escribe hype_node
    contrarian_argument:  str   # lo escribe contrarian_node
    synthesis:            str   # lo escribe arbitro_node
    report:               str   # lo escribe writer_node
    graph_html:           str   # lo escribe writer_node
    report_path:          str
    graph_path:           str
```

"Por ejemplo, `search_node` solo retorna `{"search_results": results}` — no toca el resto del estado."

```python
# app/graph.py
def search_node(state: ResearchState) -> dict:
    results = search_agent(state["question"])
    return {"search_results": results}   # ← solo modifica su campo
```

---

## BLOQUE 10 — DEMO EN VIVO

**Antes de empezar:** verificar que el servidor corre (`uvicorn app.main:app --reload`) y que el `.env` tiene `NVIDIA_API_KEY`.

**Secuencia:**

1. Abrir `http://localhost:8000` en el navegador
2. Escribir la pregunta: *"¿En qué escenarios los LLMs superan al ML tradicional?"*
3. Subir los PDFs (Attention is All You Need, Survey of LLMs, XGBoost)
4. Clic en **Iniciar debate**
5. Mostrar la terminal mientras corre — señalar cada agente cuando aparece en el log

**Narrar mientras corre:**

- *"Ahí está el Search Agent — está llamando `web_search` con DuckDuckGo"*
- *"Ahora el Reader Agent está haciendo RAG — ven el `rag_search` que llama al retriever FAISS"*
- *"Hype y Contrarian están corriendo en paralelo — dos threads del grafo al mismo tiempo"*
- *"El Árbitro recibió ambos argumentos y está sintetizando"*
- *"El Writer está en la primera iteración del loop — ahora el Critic evalúa..."*
- *"Llegó APPROVED — el reporte pasó la revisión"*

**Cuando termina:** mostrar el debate en la UI, el reporte renderizado y el grafo Pyvis.

---

## PREGUNTAS FRECUENTES

**¿Dónde está el loop agéntico?**

> `app/agents/base.py`, función `run_agent`, el `for iteration in range(1, 11)`. El LLM decide cuántas iteraciones hace según si devuelve `stop` o `tool_calls`.

**¿Cómo defines las tools?**

> JSON schema formato OpenAI en `TOOLS`, implementación Python en `TOOL_REGISTRY`. El `run_agent` recibe ambos. Ver `search_agent.py` y `reader_agent.py`.

**¿Dónde está el patrón LOOP?**

> `app/agents/writer_agent.py`, función `_generate_report`. Writer → Critic → Refiner, máximo 3 iteraciones, condición de salida `"APPROVED" in critique.upper()` verificada en Python.

**¿Cómo implementaste el paralelismo?**

> Con edges del StateGraph de LangGraph. Cuando `reader` tiene dos edges de salida (`hype` y `contrarian`), LangGraph los ejecuta en paralelo. Cuando dos edges llegan a `arbitro`, LangGraph espera a que ambos terminen.

**¿Por qué LangGraph y no ThreadPoolExecutor manual?**

> LangGraph maneja el estado compartido, el paralelismo y el merge automáticamente. Con ThreadPoolExecutor tendría que sincronizar manualmente y resolver condiciones de carrera sobre el estado.

**¿Cómo implementaste el RAG?**

> `loader.py` usa MarkItDown para convertir PDF a texto. `chunker.py` usa `RecursiveCharacterTextSplitter` con chunks de 1200 caracteres y overlap de 120. `vectorstore.py` usa `all-MiniLM-L6-v2` de HuggingFace para embeddings locales y FAISS como índice.

**¿Por qué el retriever es un closure?**

> Porque depende de los PDFs de cada sesión. Si fuera global, mezclaría documentos de distintas sesiones. El closure lo aísla al scope de cada llamada a `reader_agent`.

**¿Por qué funciones y no clases?**

> Porque así está en los notebooks del profe. Cada agente es una función que llama a `run_agent`. Una clase no agrega nada cuando una función resuelve el problema.

**¿Qué modelo usa cada agente?**

> En desarrollo todos usan `DEV_MODEL` (`nvidia/nemotron-mini-4b-instruct`) para ahorrar créditos. En producción, cada agente tiene su modelo elegido empíricamente después de 5 pruebas completas: Search y Reader usan `nemotron-mini-4b` — suficiente para búsqueda y RAG. El Hype usa `mistralai/mistral-large-3-675b-instruct-2512` — mejor calidad argumentativa con referencias reales. El Contrarian usa `minimaxai/minimax-m2.7` — estilo pragmático e industrial. El Árbitro usa `mistral-large` — mejor síntesis del ciclo. El Writer usa `minimaxai/minimax-m2.7` — también probé `mistral-large` (9.5/10, reporte perfecto) y `seed-oss-36b` (descartado — ignoraba instrucciones de idioma). La configuración actual usa minimax en el Writer porque fue la que quedó en el `.env` final.

**¿El RAG funciona con cualquier PDF?**

> Con PDFs de texto. PDFs escaneados fallan porque MarkItDown no puede extraer texto de imágenes. Está validado en `loader.py` — lanza `ValueError` si el texto está vacío.

**¿Dónde están los outputs?**

> `app/outputs/reports/` para los reportes en Markdown y `app/outputs/graphs/` para los grafos en HTML. El Writer los guarda con timestamp en el nombre.

**¿Cómo garantizas que el Árbitro sea neutral?**

> Por el system prompt. Tiene instrucciones explícitas de no tomar partido, identificar puntos válidos de ambas posiciones y señalar las tensiones irresolubles.

---

## CIERRE

"El proyecto demuestra los cuatro patrones multi-agente vistos en clase — sequential, parallel, orchestrator y loop — implementados con LangGraph sobre una pregunta real de investigación en IA. El sistema corre localmente, recibe PDFs arbitrarios y produce outputs concretos en cada ejecución: un reporte académico en Markdown y un mapa de conceptos interactivo."
