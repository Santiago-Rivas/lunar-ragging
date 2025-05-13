## Parámetros a Variar en el Grid Search

### 1. `chunk_size`

- **¿Qué es?**  
  El tamaño máximo (en caracteres) de cada fragmento (“chunk”) que genera el `RecursiveCharacterTextSplitter`.  

- **Cómo se mide**  
  Número de caracteres por chunk (p. ej. 128, 256).  

- **Rango esperado**  
  - Mínimo: 64–128 (fragmentos muy pequeños)  
  - Máximo: 512–1024 (fragmentos muy grandes)  
  - Promedio típico: 256  

- **Interpretación**  
  - Fragmentos más pequeños → más granularidad, pero mayor número de chunks y overhead.  
  - Fragmentos grandes → contexto más amplio por chunk, pero posible pérdida de foco y mayor costo de embedding.  

- **Por qué importa**  
  Afecta directamente el número de vectores en tu DB, latencia de indexado y calidad de recuperación.

---

### 2. `chunk_overlap`

- **¿Qué es?**  
  La cantidad de caracteres que comparten chunks consecutivos para mantener continuidad entre ellos.  

- **Cómo se mide**  
  Número de caracteres superpuestos (p. ej. 64, 128).  

- **Rango esperado**  
  - Mínimo: 0  
  - Máximo: ≈ chunk_size/2  
  - Promedio típico: 64–128  

- **Interpretación**  
  - Superposiciones mayores ayudan a no cortar información crítica entre fragments, a costa de más embeddings casi redundantes.  
  - Superposiciones bajas reducen la redundancia pero pueden partir conceptos a la mitad.  

- **Por qué importa**  
  Balancea redundancia vs. cobertura del texto original en vectores vecinos.

---

### 3. `k_retriever`

- **¿Qué es?**  
  Número de chunks que el retriever MMR recupera antes de re-rankear y seleccionar.  

- **Cómo se mide**  
  Entero (p. ej. 5, 10, 20).  

- **Rango esperado**  
  - Mínimo: 1–5  
  - Máximo: 50+  
  - Promedio típico: 10–20  

- **Interpretación**  
  - k pequeños → menos opciones, riesgo de perder contexto relevante.  
  - k grandes → más opciones para LLM, pero mayor latencia y ruido posible.  

- **Por qué importa**  
  Controla el trade-off entre exhaustividad y eficiencia de tu etapa de recuperación.

---

### 4. `mmr_lambda`

- **¿Qué es?**  
- MMR (Maximal Marginal Relevance) es un algoritmo de re-ranking diseñado para combinar relevancia con diversidad en la selección de documentos o fragmentos. Se usa mucho en sistemas de recuperación de información (IR) y en pipelines RAG para evitar que los primeros K resultados sean todos muy parecidos entre sí.
  Parámetro λ para Maximal Marginal Relevance (MMR): regula relevancia vs. diversidad.  

- **Cómo se mide**  
  Flotante entre 0.0 y 1.0 (p. ej. 0.2, 0.5, 0.8).  

- **Rango esperado**  
  - 0.0 → puro scorer de similitud (solo relevancia)  
  - 1.0 → puro diversidad (solo distancia respecto a lo ya seleccionado)  
  - 0.5 → balance mitad/mitad  

- **Interpretación**  
  - λ bajo → prioriza similitud estricta al query.  
  - λ alto → prioriza diversidad entre los snippets.  

- **Por qué importa**  
  Afecta la variedad de perspectivas vs. la precisión de los fragments presentados al LLM.

---

### 5. `temperature`

- **¿Qué es?**  
  Controla aleatoriedad en la generación de respuestas del LLM.  

- **Cómo se mide**  
  Flotante entre 0.0 y 1.0 (p. ej. 0.0, 0.2, 0.7).  

- **Rango esperado**  
  - 0.0 → determinista (mismo prompt→misma respuesta)  
  - 0.7 → moderada aleatoriedad  
  - 1.0 → alta aleatoriedad  

- **Interpretación**  
  - Temperaturas bajas → respuestas más “seguras” y repetibles.  
  - Temperaturas altas → respuestas más creativas pero menos consistentes.  

- **Por qué importa**  
  Balancea consistencia vs. creatividad del agente al reformular y razonar sobre los fragments recuperados.

---

Con este **grid search** podrás mapear cómo cada dimensión de configuración impacta tus métricas de rendimiento (latencia, throughput, redundancia, diversidad, faithfulness y precisión de contexto) y encontrar la combinación óptima para tu escenario.
