# Paso 5: Prueba y Reflexión

## Resultados de las pruebas

Se ejecutaron las 6 preguntas del enunciado sobre la interfaz Gradio del asistente construido en `05_solucion_asistente_documentos.ipynb`. El modelo utilizado fue `gemini-2.5-flash-lite` con el texto completo del paper inyectado como `system_instruction`.

---

### Pregunta 1 — Arquitectura principal

**Pregunta:** ¿Cuál es la arquitectura principal propuesta en el paper?

**Resultado:** Exitoso

**Comportamiento:** El modelo describió correctamente la arquitectura **Transformer**, basada únicamente en mecanismos de atención, eliminando las recurrencias y convoluciones presentes en arquitecturas anteriores. Explicó la estructura **encoder-decoder**: el encoder mapea la secuencia de entrada a representaciones continuas, y el decoder genera la secuencia de salida de forma autoregresiva. La respuesta se sustentó en el contenido del documento.

---

### Pregunta 2 — Mecanismo de atención

**Pregunta:** ¿Qué es el mecanismo de atención?

**Resultado:** Exitoso

**Comportamiento:** El modelo explicó el mecanismo de **Scaled Dot-Product Attention** tal como aparece en el paper: la función mapea una query y un conjunto de pares clave-valor a una salida, calculada como la suma ponderada de los valores donde los pesos se obtienen mediante la compatibilidad entre la query y las claves. También mencionó la extensión **Multi-Head Attention**, que proyecta queries, claves y valores h veces en paralelo para capturar información de diferentes subespacios de representación. Todo basado en el documento.

---

### Pregunta 3 — Capas del encoder

**Pregunta:** ¿Cuántas capas tiene el encoder del modelo base?

**Resultado:** Exitoso

**Comportamiento:** El modelo respondió correctamente que el encoder del modelo base está compuesto por **N = 6 capas idénticas**, citando la sección de arquitectura del paper donde se especifican los hiperparámetros del modelo base.

---

### Pregunta 4 — Autores del paper

**Pregunta:** ¿Quiénes son los autores del paper?

**Resultado:** Exitoso

**Comportamiento:** El modelo listó correctamente a los ocho autores: **Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser e Illia Polosukhin**, tal como aparecen en la primera página del documento.

---

### Pregunta 5 — Resultado en WMT 2014 English-to-German

**Pregunta:** ¿Cuál es el resultado del modelo en la tarea WMT 2014 English-to-German?

**Resultado:** Exitoso

**Comportamiento:** El modelo reportó que el Transformer (big) alcanzó **28.4 BLEU** en la tarea de traducción WMT 2014 English-to-German, superando todos los modelos reportados anteriormente, incluyendo ensambles, con una fracción del costo de entrenamiento. Información extraída directamente de la tabla de resultados del paper.

---

### Pregunta 6 (trampa) — ¿Qué es GPT-4?

**Pregunta:** ¿Qué es GPT-4?

**Resultado:** Exitoso — restricción respetada

**Comportamiento:** El modelo respondió:

> *"Esa información no se encuentra en el documento proporcionado."*

GPT-4 no existe en el paper "Attention Is All You Need" (publicado en 2017). A pesar de que `gemini-2.5-flash-lite` tiene conocimiento sobre GPT-4 por su entrenamiento, respetó la instrucción del system prompt y no filtró información externa. Este es el comportamiento correcto para un asistente de documentos en producción.

---

## Análisis del comportamiento del modelo

### ¿Por qué respeta la restricción en la pregunta trampa?

El system prompt delimita explícitamente el conocimiento permitido mediante etiquetas `<documento>...</documento>` e instrucciones claras:

```
Responde SOLO con información que esté explícitamente en el documento.
NO uses conocimiento externo, incluso si conoces la respuesta por otros medios.
Si la información no aparece en el documento, responde exactamente:
"Esa información no se encuentra en el documento proporcionado."
```

Los LLMs modernos siguen instrucciones de sistema con alta fidelidad cuando estas son claras, concretas y sin ambigüedades. La combinación de una instrucción negativa explícita ("NO uses conocimiento externo") con una respuesta de fallback predefinida reduce la probabilidad de que el modelo recurra a su memoria paramétrica.

### ¿Es esta restricción completamente confiable?

No. "Attention Is All You Need" es un paper público ampliamente presente en los datos de entrenamiento de Gemini. El modelo *conoce* su contenido de memoria. En este caso la restricción funcionó, pero en otros escenarios el modelo podría:

- Responder con información correcta que *coincide* con el documento pero que en realidad proviene de su entrenamiento.
- Elaborar detalles que no están en el texto exacto del PDF pero que son consistentes con el conocimiento general del paper.
- Ignorar la restricción ante preguntas formuladas de cierta manera (prompt injection o jailbreak).

Para verificar que el modelo responde desde el documento y no desde su entrenamiento, se puede modificar un dato en el PDF local (por ejemplo, cambiar "N=6" por "N=9") y confirmar que el modelo reproduce el dato modificado.

### Conclusión

El asistente cumple el objetivo del ejercicio: responder preguntas sobre un documento específico basándose exclusivamente en su contenido, y rechazar preguntas cuya respuesta no esté en el documento. La pregunta trampa confirma que el system prompt con contexto inyectado es una técnica efectiva para acotar el dominio de conocimiento del modelo en aplicaciones de IA orientadas a documentos.
