import pandas as pd
import matplotlib.pyplot as plt

# Carga de datos (asegúrate de haber generado 'chunk_bench_extended.csv')
df = pd.read_csv("chunk_bench_extended.csv")

# -----------------------------
# 1) Impacto de chunk_size en el tiempo de indexación
# Objetivo: Ver cómo cambia el tiempo de ingesta al variar el tamaño del fragmento.
plt.figure()
plt.plot(df["chunk_size"], df["indexing_sec"], marker="o")
plt.xlabel("Chunk Size")
plt.ylabel("Indexing Time (s)")
plt.title("Impact of Chunk Size on Indexing Time")
plt.grid(True)
plt.show()

# -----------------------------
# 2) Impacto de chunk_size en la latencia P95 de consulta
# Objetivo: Evaluar la relación entre fragmento más grande y velocidad de respuesta.
plt.figure()
plt.plot(df["chunk_size"], df["p95_latency_ms"], marker="o")
plt.xlabel("Chunk Size")
plt.ylabel("P95 Query Latency (ms)")
plt.title("Impact of Chunk Size on P95 Query Latency")
plt.grid(True)
plt.show()

# -----------------------------
# 3) Trade-off entre redundancia y diversidad
# Objetivo: Medir cómo la repetición de usuarios en chunks afecta la variedad del contexto.
plt.figure()
plt.scatter(df["redundancy"], df["diversity"])
plt.xlabel("Redundancy")
plt.ylabel("Diversity")
plt.title("Redundancy vs. Diversity")
plt.grid(True)
plt.show()

# -----------------------------
# 4) Throughput vs. número de chunks indexados
# Objetivo: Cuántos chunks por segundo podemos procesar según la fragmentación.
plt.figure()
plt.scatter(df["num_chunks"], df["throughput_cps"])
plt.xlabel("Number of Chunks")
plt.ylabel("Throughput (chunks/sec)")
plt.title("Throughput vs. Number of Chunks Indexed")
plt.grid(True)
plt.show()
