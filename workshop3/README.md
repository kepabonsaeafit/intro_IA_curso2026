# Asistente de Documentos con Gemini y Gradio

Aplicación de IA que responde preguntas sobre el paper **"Attention Is All You Need"** (Vaswani et al., 2017) usando **Gemini 2.5 Flash Lite** como modelo y **Gradio** como interfaz web.

La técnica utilizada es **inyección de contexto completo**: el texto del PDF se extrae con `pypdf` y se embebe directamente en el `system_instruction` de cada llamada a la API. Es la base conceptual de RAG (Retrieval-Augmented Generation).

## Estructura del proyecto

```
WORKSHOP3/
├── AIAYN.pdf                          # Paper "Attention Is All You Need"
├── .env                               # GEMINI_API_KEY (no incluido en el repo)
├── README.md
└── work_resources/
    ├── 03_gemini_chatbot_lab.ipynb    # Lab: SDK de Gemini con Python
    ├── 04_gemini_gradio_lab.ipynb     # Lab: Gradio + Gemini
    ├── 05_ejercicio.ipynb             # Enunciado del ejercicio
    └── 05_solucion_asistente_documentos.ipynb  # ← Solución
```

## Requisitos

- Python 3.10+
- Cuenta en [Google AI Studio](https://aistudio.google.com) para obtener la API key

### Dependencias

```bash
pip install pypdf gradio google-genai python-dotenv
```

### Configuración

Crea un archivo `.env` en la raíz del proyecto:

```
GEMINI_API_KEY="tu_key_aqui"
```

## Cómo ejecutar

1. Activa el entorno virtual:

   ```bash
   # Windows
   .venv\Scripts\activate
   ```

2. Abre `work_resources/05_solucion_asistente_documentos.ipynb` en Jupyter o VS Code.

3. Ejecuta las celdas en orden (Shift+Enter).

4. Cuando la celda del Paso 4 se ejecute, abre tu navegador en:

   ```
   http://localhost:8080
   ```

5. Escribe preguntas sobre el paper en el chat.

6. Para cerrar el servidor, ejecuta la última celda (`demo.close()`).

## Arquitectura

```
AIAYN.pdf
    │
    ▼
pypdf.PdfReader          ← extrae texto página a página
    │
    ▼
build_system_prompt()    ← embebe el texto en <documento>...</documento>
    │
    ▼
Gemini 2.5 Flash Lite    ← genera respuesta con streaming
    │
    ▼
gr.ChatInterface         ← muestra la respuesta en tiempo real
```

El texto del paper (~40K tokens) viaja en el `system_instruction` de cada llamada.
Esto funciona porque `gemini-2.5-flash-lite` tiene una ventana de contexto de **1,000,000 tokens**.

---

## Reflexión final

### 1. ¿Cuál es la limitación principal de este enfoque?

Todo el texto del documento se envía como parte del `system_instruction` en **cada request**. Esto implica:

- **Costo de tokens**: cada turno de conversación paga el costo de entrada del documento completo (aunque algunos proveedores ofrecen cache de prefijos).
- **Límite de ventana de contexto**: un documento de 1000 páginas tiene aproximadamente 500K–1M tokens, lo que saturaría la ventana de la mayoría de modelos (y en el caso de Gemini Flash, la agotaría rápidamente cuando se suma el historial de la conversación).
- **Degradación de atención** (_lost in the middle_): los LLMs tienden a prestar más atención al inicio y al final del contexto. En documentos muy largos, información del medio puede ignorarse.
- **Latencia**: procesar un contexto de 1M tokens en cada turno aumenta significativamente el tiempo de respuesta.
- **No escala a múltiples documentos**: si el corpus crece (100 PDFs, una base de conocimiento corporativa), este enfoque es inviable.

### 2. ¿Por qué existe RAG? ¿Cómo resolvería el problema anterior?

**RAG (Retrieval-Augmented Generation)** resuelve exactamente el problema de escala:

En lugar de inyectar el documento completo, RAG:

1. **Indexa** el corpus dividiéndolo en _chunks_ (fragmentos) y generando un embedding vectorial por chunk.
2. **Recupera** solo los k chunks más relevantes a la pregunta del usuario (búsqueda semántica por similitud de embeddings).
3. **Genera** la respuesta con Gemini usando únicamente esos k chunks como contexto.

Esto permite:

- Manejar corpus de **miles de documentos** sin límite de contexto.
- Reducir el **costo por request** (solo k chunks en lugar del documento completo).
- **Actualizar** el corpus sin reentrenar el modelo (solo re-indexar).
- Escalar horizontalmente la base de conocimiento.

La contrapartida es mayor complejidad de infraestructura: se necesita una base de datos vectorial (Pinecone, Weaviate, Chroma, pgvector) y un pipeline de indexación.

### 3. ¿Qué información podría "filtrarse" aunque el system prompt lo prohíba?

"Attention Is All You Need" es un paper público desde 2017 y está ampliamente presente en los datos de entrenamiento de Gemini. El modelo tiene **memoria paramétrica** sobre su contenido: puede conocer la respuesta "de memoria" incluso si el documento no estuviera en el contexto.

**Cómo verificar que el modelo responde desde el documento y no desde su entrenamiento:**

1. **Test de dato modificado**: altera un número en el PDF local (por ejemplo, cambia "6 encoder layers" por "9 encoder layers") y pregunta cuántas capas tiene el encoder. Si el modelo responde "6" (lo correcto históricamente), está usando su memoria paramétrica. Si responde "9", está leyendo el documento.

2. **Exigir cita textual**: Se pide que incluya siempre una cita literal del documento. Si la cita no aparece en el PDF, se filtraron datos del entrenamiento.

3. **Preguntar algo que no está en el paper**: la pregunta trampa _"¿Qué es GPT-4?"_ no tiene respuesta en el paper. Si el modelo responde con información correcta sobre GPT-4, está ignorando la instrucción de ceñirse al documento y usando su conocimiento general.

4. **Inyectar información falsa en el PDF**: añade una sección inventada al documento y verifica que el modelo la "crea". Si también repite esa información falsa, confirma que está leyendo el contexto. Si no la menciona, está usando su entrenamiento.

## Esta limitación es inherente a los LLMs con conocimiento previo. RAG no la elimina completamente, pero la mitiga: al recuperar chunks específicos del documento, el modelo tiene menos incentivo para "inventar" o recurrir a su entrenamiento.

## Modelo utilizado

- **gemini-2.5-flash-lite** — Free tier: 500 req/día, 1M tokens de contexto
- Documentación: [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
