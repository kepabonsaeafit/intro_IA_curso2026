# ResearchCrew — Informe Final
**Curso:** Introducción a la Inteligencia Artificial — EAFIT 2026-1  
**Estudiante:** Kevin Pabón  
**Repositorio:** [GitHub — ResearchCrew]

---

## 1. Planteamiento del Problema

### Contexto y motivación

La revisión de literatura científica es una de las tareas más costosas en tiempo dentro de la investigación académica. Un estudiante o investigador que quiere entender el estado del arte sobre un tema en IA puede enfrentarse a decenas de papers que debe leer, comparar y sintetizar manualmente, un proceso que puede tomar días.

Al mismo tiempo, la pregunta *"¿cuándo usar LLMs y cuándo usar ML tradicional?"* es una de las más relevantes y debatidas en la comunidad de IA en 2024-2025. No existe una respuesta única — depende del dominio, los datos, los recursos y los requisitos de interpretabilidad. Esta ambigüedad hace que el tema sea ideal para un sistema de debate multi-perspectiva.

### Problema

No existe una herramienta accesible que permita a un estudiante o investigador cargar papers académicos en PDF, hacer una pregunta de investigación, y obtener un análisis estructurado con múltiples perspectivas fundamentadas en los documentos cargados y en fuentes web actualizadas — todo de forma automática.

### Relevancia

El problema aborda la intersección entre dos capacidades fundamentales de la IA moderna: la recuperación de información (RAG) y la orquestación de agentes autónomos. Construir un sistema que combine ambas sobre una pregunta real de investigación demuestra aplicación práctica de los temas más avanzados del curso.

---

## 2. Objetivo General

Desarrollar un sistema multi-agente llamado **ResearchCrew** que, dado un conjunto de papers académicos en PDF y una pregunta de investigación, orqueste automáticamente un debate estructurado entre agentes con personalidades y modelos distintos, produciendo como output un reporte académico en Markdown y un mapa de conceptos interactivo generado con Pyvis.

---

## 3. Metodología

### Enfoque general

El sistema sigue un enfoque de **orquestación multi-agente con LangGraph**, combinando dos patrones principales: RAG (Retrieval-Augmented Generation) para fundamentar los argumentos en documentos reales, y agentes con personalidades distintas ejecutados en paralelo para generar perspectivas diversas sobre la misma pregunta.

### Temas del curso integrados

| Tema del curso | Aplicación en el proyecto |
|---|---|
| LLMs y Transformers (Lecture 10) | Modelos base de todos los agentes vía NVIDIA NIM |
| RAG y Embeddings (Lecture 11) | Pipeline de indexación y recuperación de papers |
| LLM-powered Agents (Lecture 12) | Loop agéntico `run_agent` con tool use |
| Multi-agente (Lecture 12) | Orquestación con LangGraph — 4 patrones implementados |
| AI Engineering | Despliegue con FastAPI, gestión de modelos via `.env` |

### Diagrama de flujo

```
┌─────────────────────────────────────────────────┐
│                   USUARIO                        │
│  Sube PDFs + escribe pregunta de investigación   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              CAPA 1 — RECOLECCIÓN                │
│                  (SEQUENTIAL)                    │
│                                                  │
│  ┌─────────────────┐    ┌─────────────────────┐  │
│  │  Search Agent   │    │    Reader Agent     │  │
│  │  (DuckDuckGo)   │───▶│  (RAG + FAISS)      │  │
│  │                 │    │                     │  │
│  │ web_search tool │    │  rag_search tool    │  │
│  └─────────────────┘    └─────────────────────┘  │
│                                  │               │
│                    shared_context▼               │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              CAPA 2 — DEBATE                     │
│                 (PARALLEL)                       │
│                                                  │
│  ┌──────────────────┐  ┌───────────────────────┐ │
│  │    El Hype       │  │    El Contrarian      │ │
│  │  (Mistral Large) │  │   (Minimax M2.7)      │ │
│  │  Pro-LLMs        │  │   Pro-ML Clásico      │ │
│  └────────┬─────────┘  └──────────┬────────────┘ │
│           │                       │              │
│           └───────────┬───────────┘              │
│                       ▼                          │
│            ┌─────────────────────┐               │
│            │     El Árbitro      │               │
│            │  (Mistral Large)    │               │
│            │  Sintetiza consenso │               │
│            └─────────────────────┘               │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              CAPA 3 — OUTPUT                     │
│                  (LOOP)                          │
│                                                  │
│  Writer genera borrador                          │
│       │                                          │
│       ▼                                          │
│  Critic evalúa → ¿APPROVED?                      │
│       │ No → Refiner corrige (máx 3 iter)        │
│       │ Sí → Entrega                             │
│       ▼                                          │
│  Reporte Markdown + Mapa Pyvis                   │
└─────────────────────────────────────────────────┘
```

### Patrones multi-agente implementados

| Patrón | Implementación | Justificación |
|---|---|---|
| **Sequential** | `search → reader` | El Reader necesita lo del Search para enriquecer el contexto |
| **Parallel** | `hype` y `contrarian` simultáneos | Son independientes — no se necesitan entre sí |
| **Orchestrator** | `StateGraph` de LangGraph | Coordina todos los nodos con estado compartido |
| **Loop** | Writer → Critic → Refiner | El reporte requiere refinamiento iterativo con criterio externo |

---

## 4. Desarrollo

### Stack técnico

| Componente | Tecnología | Rol |
|---|---|---|
| Backend | FastAPI | API REST, manejo de uploads, endpoints |
| Frontend | HTML + CSS + JS vanilla | Interfaz de usuario |
| Orquestación | LangGraph (StateGraph) | Grafo de agentes con estado compartido |
| RAG | LangChain + FAISS | Indexación y recuperación de papers |
| PDF parsing | MarkItDown | Extracción de texto de PDFs |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Vectorización local, sin API key |
| Modelos | NVIDIA NIM (OpenAI-compatible) | Proveedor de LLMs gratuito |
| Mapa de conceptos | Pyvis | Grafo interactivo HTML |
| Variables de entorno | python-dotenv | Gestión segura de API keys |

### Modelos por agente

| Agente | Modelo | Familia | Justificación |
|---|---|---|---|
| Search Agent | `nvidia/nemotron-mini-4b-instruct` | NVIDIA | Suficiente para búsqueda web |
| Reader Agent | `nvidia/nemotron-mini-4b-instruct` | NVIDIA | Suficiente para RAG |
| El Hype | `mistralai/mistral-large-3-675b-instruct-2512` | Mistral AI (Europa) | Argumentación académica con referencias |
| El Contrarian | `minimaxai/minimax-m2.7` | Minimax (China) | Estilo pragmático e industrial |
| El Árbitro | `mistralai/mistral-large-3-675b-instruct-2512` | Mistral AI (Europa) | Mejor síntesis en pruebas comparativas |
| Writer | `mistralai/mistral-large-3-675b-instruct-2512` | Mistral AI (Europa) | Único que respeta instrucciones en español |

### Arquitectura del código

```
researchcrew/
├── app/
│   ├── main.py                  # FastAPI — endpoints /research y /config
│   ├── graph.py                 # LangGraph — StateGraph y ResearchState
│   ├── agents/
│   │   ├── base.py              # run_agent — loop agéntico reutilizable
│   │   ├── search_agent.py      # web_search tool con DuckDuckGo
│   │   ├── reader_agent.py      # rag_search tool con FAISS (closure)
│   │   ├── hype_agent.py        # El Hype — Pro-LLMs
│   │   ├── contrarian_agent.py  # El Contrarian — Pro-ML clásico
│   │   ├── arbitro_agent.py     # El Árbitro — síntesis neutral
│   │   └── writer_agent.py      # Writer + Critic + Pyvis
│   ├── rag/
│   │   ├── loader.py            # MarkItDown — PDF a texto
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter
│   │   └── vectorstore.py       # FAISS + HuggingFace embeddings
│   ├── static/
│   │   └── index.html           # UI completa en un solo archivo
│   └── outputs/
│       ├── reports/             # Reportes en Markdown con timestamp
│       └── graphs/              # Grafos Pyvis en HTML con timestamp
├── papers/                      # PDFs subidos por el usuario
├── tests/
│   └── test_agents.py           # Tests por agente con DEV_MODEL
├── .env.example
├── requirements.txt
└── README.md
```

### Decisiones técnicas relevantes

**Loop agéntico (`base.py`):** La función `run_agent` es el núcleo del sistema. Implementa el patrón del notebook del profe adaptado a NVIDIA NIM: si el LLM retorna `finish_reason == "tool_calls"`, ejecuta la herramienta Python y vuelve a llamar al LLM. Si retorna `"stop"`, retorna la respuesta. Corre hasta 10 iteraciones. Incluye retry automático para errores 504 (timeout de NVIDIA NIM) con 3 intentos y espera de 10 segundos.

**Closure en el Reader Agent:** El retriever FAISS se construye como closure dentro de `reader_agent()` porque depende de los PDFs de cada sesión. Si fuera global, mezclaría documentos de distintas sesiones.

**Embeddings locales:** Se intentó usar `GoogleGenerativeAIEmbeddings` (mismo que Workshop 3), pero el free tier de Gemini tiene límite de 100 requests/minuto y el paper XGBoost genera más chunks que ese límite. Se cambió a `all-MiniLM-L6-v2` de HuggingFace — sin rate limits, sin API key, descarga única de ~90MB.

**Estado compartido (`ResearchState`):** TypedDict con un campo por agente. Cada nodo solo modifica su campo correspondiente. LangGraph hace el merge automáticamente — no hay condiciones de carrera.

---

## 5. Resultados

### Pruebas comparativas realizadas

Se realizaron 4 pruebas con distintas configuraciones de papers y modelos para determinar el stack óptimo.

#### Evolución del mapa de conceptos según número de papers

| Configuración | Papers cargados | Nodos en mapa Pyvis |
|---|---|---|
| Prueba 1 | Survey of LLMs (1 paper) | 4 nodos |
| Prueba 2 | Attention is All You Need (1 paper) | 12 nodos |
| Prueba 3 | 3 papers simultáneos | 17 nodos |

**Hallazgo:** La calidad del mapa de conceptos escala directamente con la cantidad y riqueza técnica de los papers cargados. Esto valida que el RAG está funcionando correctamente — más contexto produce outputs más ricos.

#### Calidad del debate por prueba

| Agente | Prueba 1 | Prueba 2 | Prueba 3 |
|---|---|---|---|
| El Hype | 8.5/10 | 8/10 | 9/10 |
| El Contrarian | 7.5/10 | 8.5/10 | 9/10 |
| El Árbitro | 9/10 | 9/10 | 9.5/10 |

**Hallazgo:** La calidad del debate mejora con más papers. El Árbitro con Mistral Large generó las síntesis más maduras del ciclo, identificando tensiones irresolubles entre los paradigmas (interpretabilidad vs emergencia, costo vs versatilidad, generalización vs especialización).

#### Evaluación de modelos para el Writer

| Modelo probado | Resultado | Decisión |
|---|---|---|
| `nvidia/nemotron-mini-4b-instruct` | Resumía fuentes web en vez de estructurar el reporte | Descartado |
| `bytedance/seed-oss-36b-instruct` | Generaba el reporte en inglés ignorando el system prompt | Descartado |
| `mistralai/mistral-large-3-675b-instruct-2512` | Reporte estructurado en español, APPROVED en primera iteración | **Seleccionado** |

#### Rendimiento del pipeline completo

- **Tiempo promedio de ejecución:** ~21 minutos con modelos de desarrollo
- **Tasa de éxito:** 100% en las pruebas finales (sin errores después de resolver issues de configuración)
- **Patrón LOOP:** El Critic aprobó el reporte en primera iteración en todos los casos
- **Diversidad emergente:** La combinación Mistral (europeo) + Minimax (chino) generó perspectivas genuinamente distintas sin diseñarlas manualmente — Mistral adoptó estilo académico con referencias, Minimax adoptó estilo pragmático e industrial

### Evidencia del funcionamiento

El sistema produce tres outputs concretos por ejecución:

1. **Debate estructurado** en la UI — El Hype y El Contrarian en dos columnas con sus argumentos completos, El Árbitro con la síntesis
2. **Reporte en Markdown** guardado en `app/outputs/reports/` con timestamp — incluye posición pro-LLMs, posición pro-ML clásico, síntesis y conclusiones
3. **Mapa de conceptos interactivo** en Pyvis guardado en `app/outputs/graphs/` — nodos y edges extraídos por el LLM del debate, renderizado como HTML embebido en la UI

---

## 6. Discusión

### Comparación con trabajos relacionados

**NotebookLM (Google, 2023):** El referente más cercano. NotebookLM permite cargar documentos y hacer preguntas sobre ellos con RAG. ResearchCrew va más allá al agregar debate multi-agente con personalidades distintas y mapa de conceptos generado automáticamente. La diferencia clave es que ResearchCrew no solo responde preguntas — genera perspectivas en tensión, lo que es más útil para investigación crítica.

**AutoGen (Microsoft, 2023):** Framework multi-agente que permite definir conversaciones entre agentes. ResearchCrew comparte el paradigma pero con una arquitectura más simple y enfocada: un único dominio de debate, modelos heterogéneos por diseño, y output estructurado en vez de conversación abierta.

**LangGraph (LangChain, 2024):** ResearchCrew usa LangGraph como orquestador, aprovechando su manejo de estado compartido y paralelismo nativo. La diferencia es que ResearchCrew define un grafo fijo con roles especializados, no un agente genérico.

### Limitaciones identificadas

**Alucinaciones de citas:** En la Prueba 2, El Hype generó referencias académicas fabricadas (`Cutler et al. 2021`, `Kaplinsky 2020`) que no provienen del paper cargado. Este es un problema conocido de los LLMs — generan citas que suenan plausibles pero no son verificables. La solución sería restringir el system prompt para citar únicamente el contenido recuperado por el RAG.

**Tiempo de ejecución:** ~21 minutos con modelos grandes hace que el sistema sea lento para uso interactivo. En producción se podría optimizar con modelos más pequeños para los agentes de debate secundarios o con streaming de resultados.

**PDFs escaneados:** El sistema no soporta PDFs escaneados porque MarkItDown no puede extraer texto de imágenes. Sería necesario agregar OCR (por ejemplo, con Tesseract) para manejar este caso.

**Rate limits de embeddings:** El free tier de Gemini Embeddings tiene límite de 100 requests/minuto, insuficiente para papers largos. Se resolvió con embeddings locales de HuggingFace, pero en un sistema productivo se recomendaría usar un modelo de embeddings con mayor cuota.

### Estado del arte en sistemas multi-agente

El campo de los LLM-powered agents ha evolucionado rápidamente en 2024-2025. Los enfoques principales son:

- **ReAct (Yao et al., 2022):** Razonamiento + acción intercalados. ResearchCrew implementa este patrón en `run_agent` — el LLM razona y decide cuándo llamar tools.
- **Multi-agent debate (Du et al., 2023):** Múltiples LLMs debatiendo para llegar a mejores respuestas que un solo modelo. ResearchCrew extiende este enfoque usando modelos heterogéneos con personalidades distintas.
- **RAG (Lewis et al., 2020):** Recuperación de documentos para fundamentar respuestas. ResearchCrew usa RAG para garantizar que los argumentos estén anclados en evidencia real de los papers cargados.

La contribución específica de ResearchCrew es la combinación de estas tres ideas — ReAct + debate multi-modelo + RAG — en un sistema orientado a la revisión de literatura científica con output estructurado.

---

*Proyecto Final — Introducción a la Inteligencia Artificial, EAFIT 2026-1*
