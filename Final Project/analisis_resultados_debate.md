# Análisis de Resultados — Debate Multi-Agente ResearchCrew

**Proyecto:** ResearchCrew — Sistema multi-agente para debate académico  
**Pregunta de investigación:** ¿En qué escenarios los LLMs superan al ML tradicional y en cuáles no?  
**Papers procesados:** Survey of Large Language Models  
**Modelos usados:** Mistral Large 675B (El Hype), Minimax M2.7 (El Contrarian), Mistral Large 675B (El Árbitro)

---

## 1. Evaluación por Agente

### El Hype (Mistral Large 675B — Pro-LLMs)

**Fortalezas del argumento:**

- Fundamentación técnica sólida: cita arquitectura transformer, self-attention, scaling laws con referencias reales (Kaplan et al., Wei et al., Brown et al.)
- Introduce conceptos avanzados correctamente: emergent capabilities, in-context learning, few-shot/zero-shot learning
- Reconoce limitaciones sin abandonar su posición — admite debilidades en datos tabulares e interpretabilidad, pero los enmarca como trade-offs, no como fallos
- Referencia directa al paper cargado (Survey of LLMs) para respaldar argumentos

**Debilidades detectadas:**

- Longitud excesiva — tres párrafos densos que en una presentación real serían difíciles de seguir
- Algunas afirmaciones superlativas sin matiz: "supremacía indiscutible" es demasiado fuerte para un debate académico
- La referencia a ScienceDirect sobre cáncer de pulmón no proviene del paper cargado, sino de la web search — potencialmente menos confiable

**Calidad estimada:** 8.5/10 — argumento riguroso y bien estructurado, con leve tendencia a la hipérbole

---

### El Contrarian (Minimax M2.7 — Pro-ML Clásico)

**Fortalezas del argumento:**

- Estilo combativo y directo — usa lenguaje pragmático que contrasta perfectamente con El Hype
- Argumentos concretos con datos: "80-90% de los casos de uso empresarial", "microsegundos vs segundos", "megabytes vs gigabytes"
- Introduce regulación real (GDPR) como barrera de adopción para LLMs — muy relevante para la industria
- La estructura por secciones (Interpretabilidad, Eficiencia, Robustez) facilita la lectura y la defensa en demo

**Debilidades detectadas:**

- Mezcla idiomas involuntariamente: aparecen palabras en ruso (`ограниченных`) y portugués (`poucos`) — artefacto del modelo Minimax que fue entrenado multilingüe
- Algunas afirmaciones sin fuente directa: el "80-90% de datos tabulares" es una estimación de la industria, no del paper
- El cierre es demasiado absoluto: "la respuesta pragmática favorece al ML tradicional" ignora los escenarios donde los LLMs son claramente superiores

**Calidad estimada:** 7.5/10 — argumentación convincente y pragmática, penalizada por los artefactos multilingües y la falta de balance en la conclusión

---

### El Árbitro (Mistral Large 675B — Consenso)

**Fortalezas de la síntesis:**

- Estructura impecable con tres secciones: posición pro-LLM, posición pro-ML, y tensiones irresolubles
- Identifica correctamente las tres tensiones fundamentales: escalabilidad vs eficiencia, generalización vs robustez, automatización vs supervisión
- La conclusión de "coexistencia asimétrica" es precisa y matizada — reconoce que ambos paradigmas tienen hegemonía en dominios distintos
- Propone la integración estratégica como solución óptima, con ejemplo concreto (LLMs para interpretación + ML para precisión)

**Debilidades detectadas:**

- Longitud muy alta para una síntesis — cuatro párrafos extensos donde dos serían suficientes
- Repite algunos argumentos que ya presentaron los debatientes en vez de solo sintetizar
- No cuantifica las áreas de acuerdo vs desacuerdo — sería más útil decir "coinciden en 3 de 5 puntos"

**Calidad estimada:** 9/10 — la mejor pieza del debate, con síntesis equilibrada y conclusiones matizadas

---

## 2. Análisis del Debate como Sistema

### Diversidad de perspectivas

El uso de modelos de diferentes familias (Mistral vs Minimax) generó perspectivas genuinamente distintas. El Hype con Mistral Large produjo argumentación académica con referencias, mientras que el Contrarian con Minimax adoptó un tono más directo e industrial. Esta diversidad no fue diseñada manualmente — emerge naturalmente de las diferencias entre los modelos.

### Calidad del contexto compartido

Los agentes de Capa 1 (Search + Reader) proporcionaron contexto relevante pero limitado. El SearchAgent encontró fuentes reales (ScienceDirect, LinkedIn, NextTechToday) y el ReaderAgent extrajo citas del Survey of LLMs. Sin embargo, la búsqueda web tendió a fuentes divulgativas en vez de papers académicos — esto se refleja en que los debatientes a veces citan blogs en vez de investigación primaria.

### Mapa de conceptos

El grafo Pyvis generado tiene 4 nodos (LLMs, ML, trade-offs, ecosystem) con edges que representan relaciones claras. Es funcional pero podría ser más rico — con los modelos grandes y más papers cargados, el ConceptExtractor debería generar grafos con 8-12 nodos que mapeen conceptos como interpretabilidad, scaling laws, datos estructurados, etc.

### Rendimiento

- **Tiempo total:** ~21 minutos con modelos de desarrollo
- **Tasa de éxito:** pipeline completo sin errores (después de resolver issues de configuración)
- **El Critic aprobó el reporte en primera iteración** — el loop de refinamiento no fue necesario

---

## 3. Áreas de Mejora Identificadas

| Área                      | Problema                                                 | Solución propuesta                                               |
| ------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| Writer con modelo pequeño | Genera resumen de fuentes en vez de reporte estructurado | Usar `mistral-large` para el Writer en la demo final             |
| Artefactos multilingües   | El Contrarian mezcla idiomas (ruso, portugués)           | Cambiar a `mistral-nemotron` como modelo estable sin deprecación |
| Profundidad del RAG       | ReaderAgent extrae solo un fragmento del paper           | Subir 3+ papers para dar más material al sistema                 |
| Mapa de conceptos         | Solo 4 nodos — poco detallado                            | El modelo grande debería extraer más conceptos                   |
| Reporte del Writer        | Con modelo pequeño no sigue la estructura esperada       | Probar con modelo grande — el system prompt es correcto          |

---

## 4. Segunda Prueba — Paper: Attention is All You Need

**Configuración:** Un solo PDF (Attention is All You Need, Vaswani et al. 2017) + mismos modelos  
**Diferencia clave:** Paper técnico de arquitectura transformer vs. Survey general de LLMs

### Comparación entre pruebas

| Aspecto                      | Prueba 1 (Survey of LLMs)    | Prueba 2 (Attention is All You Need)                   |
| ---------------------------- | ---------------------------- | ------------------------------------------------------ |
| Paper cargado                | Survey general               | Paper técnico fundacional                              |
| Nodos en mapa Pyvis          | 4 nodos simples              | 12 nodos detallados                                    |
| Calidad del RAG              | Cita correctamente el survey | Extrae conceptos técnicos específicos                  |
| Argumentación del Hype       | Referencias genéricas a LLMs | Cita self-attention, scaling laws con mayor precisión  |
| Argumentación del Contrarian | Pragmática e industrial      | Más enfocada en el trilema eficiencia-datos-fiabilidad |

### El Hype (Prueba 2)

Con el paper de Attention is All You Need como contexto, El Hype generó un argumento notablemente más técnico con referencia directa al mecanismo de self-attention, scaling laws y emergent capabilities. Sin embargo, aparecen referencias inventadas (`Cutler et al. 2021`, `Kaplinsky 2020`) que no provienen del paper cargado — artefacto conocido como alucinación de citas. Esta es un área de mejora crítica: el Reader Agent debería restringir más las fuentes al contenido del paper.

**Calidad estimada:** 8/10 — técnicamente más preciso que la prueba 1, penalizado por citas fabricadas.

### El Contrarian (Prueba 2)

Minimax M2.7 adoptó un tono más estructurado, organizando su argumento en tres bloques: interpretabilidad, el trilema eficiencia-datos-fiabilidad, y la distinción entre "demo impresionante" y "mejor solución". La frase _"la comunidad ha confundido 'impresionante demo' con 'mejor solución'"_ es el argumento más original y contundente del debate completo. Notablemente no aparecieron artefactos multilingües, lo que sugiere que el contexto en inglés del paper estabiliza el idioma interno del modelo.

**Calidad estimada:** 8.5/10 — mejor que la prueba 1, más organizado y sin artefactos.

### Mapa de conceptos (Prueba 2)

El grafo mejoró significativamente: 12 nodos vs 4 en la primera prueba. Los nodos incluyen conceptos técnicos del paper (adaptabilidad lingüística, generación de texto, calibración rigurosa, capacidad estadística) organizados en dos clusters — uno alrededor de LLMs y otro alrededor de ML tradicional — conectados por `machine learning` como nodo puente. La calidad del mapa escala directamente con la riqueza técnica del paper cargado.

### Conclusión de la comparación

Cargar papers más técnicos y específicos mejora la calidad del debate en todos los agentes. El sistema se beneficia de documentos con vocabulario técnico denso porque el ReaderAgent extrae conceptos más ricos para el contexto compartido. Para la demo final se recomienda cargar los tres papers simultáneamente.

---

## 5. Tercera Prueba — 3 Papers Simultáneos

**Configuración:** Attention is All You Need + Survey of LLMs + XGBoost + mismos modelos  
**Diferencia clave:** Mayor volumen de contexto para el RAG — tres perspectivas distintas en el corpus

### El Hype (Prueba 3)

El argumento más sólido del ciclo completo. Con tres papers, El Hype pudo citar directamente el Survey of LLMs y el paper de XGBoost para reconocer limitaciones sin debilitar su posición. La estructura en tres párrafos es la mejor del ciclo: superioridad en NLP → ventaja en multifuncionalidad → reconocimiento honesto de debilidades. Notable el uso del paper XGBoost como evidencia _en contra_ de la posición contraria — un movimiento retórico sofisticado que solo es posible porque el Reader Agent indexó ese paper.

**Calidad estimada:** 9/10 — el mejor argumento del Hype en las tres pruebas.

### El Contrarian (Prueba 3)

Minimax M2.7 hizo algo inteligente: citar directamente el Survey of LLMs _en contra_ de los LLMs. La frase _"La [Survey of LLMs] reconoce abiertamente que los LLMs actuales pueden carecer de interpretabilidad..."_ es exactamente el tipo de argumento que un debatiente experto usaría — usar las fuentes del oponente para fortalecer la posición propia. El cierre con la pregunta invertida (_"¿por qué adoptar tecnología masiva y opaca cuando los métodos tradicionales resuelven el problema?"_) es retóricamente efectivo.

**Calidad estimada:** 9/10 — el mejor argumento del Contrarian en las tres pruebas.

### El Árbitro (Prueba 3)

Con más contexto disponible, el Árbitro generó la síntesis más estructurada del ciclo. Introduce el concepto de "complementariedad asimétrica" y cierra con _"la coexistencia, no la dominancia, definirá el futuro de la inteligencia artificial"_ — una conclusión académicamente sólida. También identifica correctamente la paradoja: los LLMs priorizan generalización sobre transparencia, el ML tradicional hace lo contrario, y ninguno es absolutamente superior.

**Calidad estimada:** 9.5/10 — mejor síntesis del ciclo completo.

### Mapa de conceptos (Prueba 3)

El grafo con 3 papers alcanzó ~17 nodos, el más rico del ciclo. Clusters claramente diferenciados: LLMs (in-context learning, razonamiento contextual, adaptabilidad, scaling laws, sesgos, opacidad) y ML tradicional (herramienta especializada y confiable, escenarios críticos). Los nodos de limitaciones de cada paradigma aparecen como satélites del nodo principal correspondiente — estructura que refleja fielmente el debate.

### El Writer con nemotron-mini (Prueba 3)

El reporte generado muestra claramente las limitaciones del modelo pequeño: mezcla idiomas (`rhesus`, `deckung`), incluye referencias inventadas (`NLP-BCE-CAAM-18`), y la síntesis pierde coherencia en el segundo párrafo. Esto confirma que el Writer _necesita_ un modelo grande para generar el entregable final. El contenido del debate fue excelente — el problema es únicamente el modelo de síntesis.

---

## 6. Comparación de las Tres Pruebas

| Métrica                 | Prueba 1 (1 paper — Survey) | Prueba 2 (1 paper — Attention) | Prueba 3 (3 papers)   |
| ----------------------- | --------------------------- | ------------------------------ | --------------------- |
| Nodos en mapa Pyvis     | 4                           | 12                             | ~17                   |
| Calidad del Hype        | 8.5/10                      | 8/10                           | 9/10                  |
| Calidad del Contrarian  | 7.5/10                      | 8.5/10                         | 9/10                  |
| Calidad del Árbitro     | 9/10                        | 9/10                           | 9.5/10                |
| Calidad del Writer      | 5/10 (modelo pequeño)       | N/A                            | 4/10 (modelo pequeño) |
| Artefactos multilingües | Sí (Contrarian)             | No                             | No                    |
| Citas fabricadas        | No                          | Sí (Hype)                      | No                    |

**Conclusión:** La calidad del debate escala con el número y calidad de papers cargados. El punto de quiebre es el Writer — con modelo pequeño el reporte final no es presentable. Para la demo se requiere `mistral-large` en el Writer.

---

## 8. Prueba A — Writer: seed-oss-36b (con fix de idioma)

**Configuración:** 3 papers + HYPE/ARBITRO: mistral-large + CONTRARIAN: minimax-m2.7 + WRITER: seed-oss-36b + fix "Responde SIEMPRE en español"

### Debate (Hype + Contrarian + Árbitro) — Mejor del ciclo

Con los modelos grandes en los tres agentes de debate, la calidad alcanzó su punto más alto. El Contrarian estructuró su argumento en cuatro bloques precisos (costo, datos estructurados, interpretabilidad, zero-shot sobrevalorado) con datos concretos y citas directas del Survey of LLMs _en contra_ de los LLMs. El Hype respondió con referencias académicas reales (Brown et al. 2020, Kaplan et al. 2020, Radford et al. 2019) y reconoció las limitaciones sin perder su posición. El Árbitro generó la síntesis más madura del ciclo completo, identificando tres tensiones irresolubles con nombres precisos: interpretabilidad vs emergencia, costo vs versatilidad, generalización vs especialización.

**Calidad del debate: 9.5/10** — el mejor del ciclo.

### Writer (seed-oss-36b) — Descartado definitivamente

Dos problemas críticos que persisten incluso con el fix de idioma:

**Problema 1 — Idioma:** seed-oss-36b ignoró "Responde SIEMPRE en español" y generó el reporte completo en inglés. El fix fue efectivo con mistral-large pero completamente ignorado por este modelo, confirmando que el problema es estructural.

**Problema 2 — ConceptExtractor incoherente:** Los nodos generados incluyen "zoom", "space", "context", "empecé", "laserpunto", "meridianos" — conceptos sin relación con el debate. El modelo perdió coherencia en extracción de JSON estructurado, produciendo un grafo Pyvis sin valor académico.

**Veredicto: seed-oss-36b descartado para el Writer.** El modelo no respeta instrucciones de idioma bajo ninguna configuración y pierde coherencia en tareas estructuradas.

---

## 9. Prueba B — Writer: mistral-large-3-675b-instruct-2512

**Configuración:** 3 papers + todos los modelos grandes + WRITER: mistral-large

### Writer (mistral-large) — Mejor del ciclo

- Reporte en español perfecto ✅
- Estructura completa con todas las secciones obligatorias ✅
- Referencias académicas reales: Vaswani et al. (2017), Kaplan et al. (2020), Wei et al. (2022), Bommasani et al. (2021), Rudin (2019), Topol (2019) ✅
- Critic aprobó en primera iteración con feedback constructivo (sugirió tabla comparativa y más ejemplos empíricos) ✅

**Calidad Writer: 9.5/10** — mejor Writer del ciclo completo.

### ConceptExtractor (mistral-large) — Sólido

8 nodos coherentes: LLMs, ML tradicional, Transformer, Scaling laws, Interpretabilidad, Eficiencia computacional, Generalización, Hibridación. Edges con relaciones precisas y académicamente válidas. Auto-agrega Opacidad y Especialización referenciados en edges (fix Pyvis activo).

**Calidad ConceptExtractor: 9/10**

---

## 10. Prueba C — Writer: minimax-m2.7

**Configuración:** 3 papers + todos los modelos grandes + WRITER: minimax-m2.7

### Writer (minimax-m2.7) — Funcional con artefactos

- Español como idioma principal ✅
- Estructura correcta ✅
- Pero aparecen artefactos multilingües persistentes:
  - `"struggles to replicate"` — inglés en medio del texto
  - `"peu fiable"` — francés (debería ser "poco fiable")
  - `"capturarctx contextuales"` — texto garbled
  - `"generalización broad"` — mezcla español-inglés

**Calidad Writer: 7/10** — penalizado por artefactos multilingües.

### Critic (minimax-m2.7) — Sorprendentemente preciso

El mismo modelo que cometió los errores los identificó correctamente: detectó "struggles" y "peu fiable" y dio APPROVED con observaciones concretas. Incluye tabla comparativa de argumentos. Muestra que minimax-m2.7 tiene capacidad crítica pero inconsistencia generativa.

### ConceptExtractor (minimax-m2.7) — Sólido

8 nodos coherentes: LLMs, ML tradicional, Escalabilidad, Interpretabilidad, Robustez, Flexibilidad, Precisión, Generalización. Edges con relaciones claras. Un edge duplicado menor.

**Calidad ConceptExtractor: 8.5/10**

---

## 11. Comparativa final de Writers

| Modelo             | Calidad reporte | Idioma                       | ConceptExtractor | Veredicto          |
| ------------------ | --------------- | ---------------------------- | ---------------- | ------------------ |
| `mistral-large`    | 9.5/10          | Español perfecto             | 9/10             | ✅ **ELEGIDO**     |
| `minimax-m2.7`     | 7/10            | Español con artefactos       | 8.5/10           | Fallback aceptable |
| `nemotron-mini-4b` | 4/10            | Mezcla idiomas, garbled      | 3/10             | ❌ Descartado      |
| `seed-oss-36b`     | 3/10            | Inglés, ignora instrucciones | 2/10             | ❌ Descartado      |

---

## 12. Configuración final para la demo

| Agente          | Modelo                                         | Razón                                              |
| --------------- | ---------------------------------------------- | -------------------------------------------------- |
| Search + Reader | `nvidia/nemotron-mini-4b-instruct`             | Suficiente para búsqueda y RAG                     |
| El Hype         | `mistralai/mistral-large-3-675b-instruct-2512` | Mejor calidad argumentativa, referencias reales    |
| El Contrarian   | `minimaxai/minimax-m2.7`                       | Estilo pragmático e industrial, estructura clara   |
| El Árbitro      | `mistralai/mistral-large-3-675b-instruct-2512` | Mejor síntesis del ciclo, tensiones bien nombradas |
| Writer          | `mistralai/mistral-large-3-675b-instruct-2512` | Español perfecto, referencias académicas, 9.5/10   |

**Para la demo:** cargar los 3 papers simultáneamente (Attention is All You Need + Survey of LLMs + XGBoost). Esta combinación produce el mapa Pyvis más rico (~17 nodos) y el debate de mayor calidad.

**Tiempo estimado:** 25-30 minutos con esta configuración completa.

---

## 13. Conclusión General

El sistema ResearchCrew demostró ser funcional de punta a punta: sube papers, busca en la web, genera un debate estructurado entre agentes con personalidades y modelos distintos, sintetiza posiciones y genera un mapa de conceptos interactivo. Los resultados del debate son académicamente válidos, con argumentos fundamentados en evidencia de los papers cargados y fuentes web reales. Las principales limitaciones son operativas (timeouts, modelos en deprecación, artefactos multilingües en modelos de entrenamiento multilingüe) más que arquitectónicas — la orquestación con LangGraph y el patrón de agentes funcionó exactamente como se diseñó.

**Estado para la demo:** listo. Configuración de modelos validada empíricamente en 5 pruebas completas.
