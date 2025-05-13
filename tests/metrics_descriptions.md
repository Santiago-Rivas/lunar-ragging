# Métricas de Benchmarking para Connection RAG

## Faithfulness (`faithfulness`)

**Qué es:**
Cuantifica la fracción de la respuesta generada por el LLM que está realmente respaldada por los snippets de contexto recuperados, evitando “alucinaciones” o inventos.

**Cómo se mide (con RAGAS):**

1. **Montaje del dataset de evaluación**

   ```python
   sample = {
     "question": ["Suggest matches for user 1"],
     "answer":   [raw_answer],
     "contexts": [[d.page_content for d in all_docs[:10]]],
   }
   ds = Dataset.from_dict(sample)
   ```
2. **Invocar la métrica Faithfulness**

   ```python
   df_eval = evaluate(dataset=ds, metrics=[Faithfulness()]).to_pandas()
   faith_score = df_eval.loc[0, "faithfulness"]
   ```
3. **Cálculo interno (resumen):**

   * La respuesta se divide en **fragmentos** (usualmente oraciones o spans).
   * Para cada fragmento, RAGAS verifica si existe **evidencia textual o semántica** en alguno de los contextos.
   * El score es la **proporción** de fragmentos que pudieron alinearse exitosamente con los snippets:

     $$
       \text{Faithfulness} = \frac{\#\{\text{fragmentos respaldados}\}}{\#\{\text{fragmentos totales en la respuesta}\}}
     $$

**Rango de valores esperado:**

* **0.0**: ninguna parte de la respuesta coincide con los contextos
* **1.0**: la respuesta entera está fundamentada en los snippets
* **Promedio típico**: 0.5 – 0.9 (dependiendo de la complejidad de la respuesta y calidad del contexto)

**Interpretación:**

* **Alto (>0.8)**: la respuesta está casi íntegramente basada en documentos recuperados → muy fiable.
* **Medio (0.5–0.8)**: mezcla de contenido extraído y algo de parafraseo propio del LLM.
* **Bajo (<0.5)**: riesgo de que el modelo “invente” información no presente en el contexto.

**Por qué es importante:**

* Garantiza **transparencia**: cada afirmación en la respuesta puede rastrearse hasta un fragmento concreto.
* Evita la **propagación de errores**: si el LLM se desvía, credibles sistemas RAG deben advertir al usuario sobre posibles invenciones.
* Aumenta la **confianza**: un alto faithfulness comunica que el sistema no está ampliando datos no verificados.

## Context Precision (`context_precision`)

**Qué es:**  
Mide la “precisión” con la que el retriever ordena y devuelve los fragmentos de contexto más alineados con la consulta (aquí, el dossier del usuario). En lugar de un precision@k clásico que requiere etiquetas, usamos un proxy basado en similitud semántica.

**Cómo se mide (en el código):**  
1. **Embed de la query**  
```python
   query_emb = rag.embeddings.embed_query(target_dossier)
```

2. **Embed de los top-K contextos**

```python
   top_docs = all_docs[:DIVERSITY_K]
   doc_embs  = rag.embeddings.embed_documents([d.page_content for d in top_docs])
```
3. **Similitud coseno**

```python
   sims = cosine_similarity([query_emb], doc_embs)[0]
```
4. **Umbral de relevancia**
   Definimos un umbral (por ejemplo `THRESH=0.75`) y contamos cuántos embeddings superan esa similitud:

```python
   n_relevant = sum(1 for s in sims if s >= THRESH)
   context_precision = n_relevant / len(sims)
```

**Rango de valores esperado:**

* **0.0**: ningún contexto recuperado está realmente alineado con la query
* **1.0**: todos los contextos recuperados superan el umbral de similitud
* **Promedio típico**: 0.6 – 0.9 (según densidad de tu corpus)

**Interpretación:**

* Valores cercanos a 1.0 indican que la mayoría de los fragments devueltos coinciden fuertemente con lo que la consulta semántica (“dossier”) demanda.
* Valores bajos (< 0.5) sugieren que el retriever está trayendo información más tangencial, lo que puede degradar tanto la relevancia de la respuesta como su fidelidad.

**Por qué es importante:**

* Asegura que el LLM reciba **contexto de alta calidad**, maximizando la probabilidad de generar respuestas correctas.
* Es un primer filtro para diagnosticar problemas de recuperación antes de llegar al LLM: si la precisión de contexto es baja, no tiene sentido optimizar prompts o modelos de chat.

---


## Redundancia de chunks (`redundancy`)

**Qué es:**  
Grado de repetición de `user_id` en los top 10 chunks recuperados.

**Cómo se mide:**  
```text
redundancy = 1 – (número de IDs únicos) / (total de chunks)

```

## Número de chunks indexados (`num_chunks`)

**Qué es:**  
Cantidad total de fragmentos de texto (“chunks”) que el pipeline ha generado y almacenado en ChromaDB tras la ingesta.

**Cómo se mide:**  
Se consulta `len(rag.collection.get(include=[])["ids"])` justo después de indexar.

**Interpretación:**  
Un número muy alto indica documentos muy fragmentados o superposición excesiva; muy bajo puede significar demasiada agregación y pérdida de granularidad.

**Importancia:**  
Equilibra **resolución semántica** (más chunks → búsquedas más finas) vs. **coste** (más vectores → más almacenamiento y latencia).

---

## Tiempo de indexación (`indexing_sec`)

**Qué es:**  
Segundos empleados en leer todos los dossiers/HTML y generar embeddings para cada chunk.

**Cómo se mide:**  
`t1 - t0` alrededor de `ingest_from_dirs(...)` con `time.perf_counter()`.

**Interpretación:**  
Cuanto menor, más rápido está tu pipeline de ingesta. Si sube drásticamente al aumentar chunks, revisar batch size o chunk size.

**Importancia:**  
Determina la **velocidad de actualización** del índice, crucial para casos de ingesta incremental o múltiples eventos.

---

## Throughput de indexación (`throughput_cps`)

**Qué es:**  
Chunks indexados por segundo.

**Cómo se mide:**  
`num_chunks / indexing_sec`.

**Interpretación:**  
Mayor throughput significa mejor escalabilidad. Valores bajos pueden indicar cuello de botella en llamadas a OpenAI o escritura en disco.

**Importancia:**  
Permite dimensionar hardware y decidir si paralelizar o reducir llamadas a embeddings.

---

## Uso de memoria antes/después de indexar (`mem_before_idx`, `mem_after_idx`)

**Qué es:**  
Memoria residente (RSS, en MB) del proceso antes y después de la indexación.

**Cómo se mide:**  
`psutil.Process().memory_info().rss / 1024**2` antes y después de `ingest_from_dirs`.

**Interpretación:**  
Incrementos grandes indican over-caching de vectores o retención excesiva de datos en memoria.

**Importancia:**  
Esencial para asegurar que tu aplicación pueda escalar sin quedarse sin RAM, especialmente en entornos cloud o contenedores.

---

## Uso de memoria antes/después de las queries (`mem_before_q`, `mem_after_q`)

**Qué es:**  
Memoria residente antes y después de ejecutar varias llamadas `suggest_connections`.

**Cómo se mide:**  
Mismo método `psutil` alrededor del bucle de queries.

**Interpretación:**  
Demasiada ganancia de memoria por query puede señalar fugas o acumulación de cachés LLM.

**Importancia:**  
Mantener la memoria estable durante la fase “online” garantiza rendimiento y evita OOM en producción.

---

## Latencia P95 de consulta (`p95_latency_ms`)

**Qué es:**  
Percentil 95 de la distribución de tiempos de respuesta (en ms) de `suggest_connections`.

**Cómo se mide:**  
Ordenar 20 latencias y tomar el valor en posición `0.95 * N`.

**Interpretación:**  
P95 refleja la “peor” latencia que ven el 95 % de los usuarios. Un P95 alto puede perjudicar la experiencia.

**Importancia:**  
Clave para SLAs; la UX perceptible suele juzgarse en función de percentiles altos (P90–P99).

---


