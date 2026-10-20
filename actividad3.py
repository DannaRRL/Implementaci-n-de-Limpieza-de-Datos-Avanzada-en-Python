
import pandas as pd
import numpy as np
import matplotlib

# Permite guardar gráficos en Ubuntu sin interfaz gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# ==========================================
# PRÁCTICA 3: LIMPIEZA AVANZADA DE DATOS
# ==========================================

# 1. GENERACIÓN DEL DATASET ORIGINAL
data = {
    "edad": [
        25, 30, np.nan, 45, 22,
        120, 35, 28, np.nan, 40
    ],
    "ingreso_mensual": [
        15000, 22000, 18000, np.nan, 12000,
        950000, 25000, 21000, 19000, np.nan
    ],
    "scoring_credito": [
        650, 700, 680, 710, 590,
        800, 670, 640, 620, 730
    ]
}

df_original = pd.DataFrame(data)

print("\n=== DATASET ORIGINAL ===")
print(df_original)

print("\n=== VALORES FALTANTES ORIGINALES ===")
print(df_original.isnull().sum())

# 2. CÁLCULO DEL RANGO INTERCUARTÍLICO
Q1 = df_original["ingreso_mensual"].quantile(0.25)
Q3 = df_original["ingreso_mensual"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print("\n=== ANÁLISIS IQR ===")
print(f"Primer cuartil (Q1): {Q1:,.2f}")
print(f"Tercer cuartil (Q3): {Q3:,.2f}")
print(f"Rango intercuartílico: {IQR:,.2f}")
print(f"Límite inferior: {limite_inferior:,.2f}")
print(f"Límite superior: {limite_superior:,.2f}")

# 3. DETECCIÓN Y TRATAMIENTO DE OUTLIERS
df_tratado = df_original.copy()

outliers_ingreso = (
    (df_tratado["ingreso_mensual"] < limite_inferior) |
    (df_tratado["ingreso_mensual"] > limite_superior)
)

outliers_edad = df_tratado["edad"] > 100

print("\n=== REGISTROS ANÓMALOS ===")
print(df_original.loc[outliers_ingreso | outliers_edad])

df_tratado.loc[
    outliers_ingreso, "ingreso_mensual"
] = np.nan

df_tratado.loc[
    outliers_edad, "edad"
] = np.nan

print("\n=== DATASET DESPUÉS DE TRATAR OUTLIERS ===")
print(df_tratado)

print("\n=== VALORES FALTANTES POR IMPUTAR ===")
print(df_tratado.isnull().sum())

# 4. IMPUTACIÓN CON KNN (K=2)
imputer_knn2 = KNNImputer(n_neighbors=2)

df_knn2 = pd.DataFrame(
    imputer_knn2.fit_transform(df_tratado),
    columns=df_tratado.columns
)

print("\n=== DATASET LIMPIO CON KNN (K=2) ===")
print(df_knn2.round(2))

# 5. COMPARACIÓN CON KNN (K=3)
imputer_knn3 = KNNImputer(n_neighbors=3)

df_knn3 = pd.DataFrame(
    imputer_knn3.fit_transform(df_tratado),
    columns=df_tratado.columns
)

# 6. IMPUTACIÓN CON MEDIA Y MEDIANA
df_media = df_tratado.fillna(df_tratado.mean())
df_mediana = df_tratado.fillna(df_tratado.median())

# Comparar únicamente los valores que faltaban
comparacion = []

for columna in df_tratado.columns:
    indices = df_tratado.index[
        df_tratado[columna].isna()
    ]

    for indice in indices:
        comparacion.append({
            "registro": indice,
            "columna": columna,
            "KNN_k2": df_knn2.loc[indice, columna],
            "KNN_k3": df_knn3.loc[indice, columna],
            "media": df_media.loc[indice, columna],
            "mediana": df_mediana.loc[indice, columna]
        })

df_comparacion = pd.DataFrame(comparacion)

print("\n=== COMPARACIÓN DE MÉTODOS ===")
print(df_comparacion.round(2).to_string(index=False))

# 7. GRÁFICOS ANTES Y DESPUÉS
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(
    df_original["ingreso_mensual"].dropna()
)
axes[0].set_title("Ingresos antes de la limpieza")
axes[0].set_ylabel("Ingreso mensual")

axes[1].boxplot(
    df_knn2["ingreso_mensual"].dropna()
)
axes[1].set_title("Ingresos después de KNN (k=2)")
axes[1].set_ylabel("Ingreso mensual")

plt.tight_layout()
plt.savefig("comparacion_boxplots.png", dpi=200)
plt.close()

# 8. GRÁFICO DE VALORES FALTANTES
faltantes_antes = df_original.isnull().sum()
faltantes_despues = df_knn2.isnull().sum()

comparacion_nulos = pd.DataFrame({
    "Original": faltantes_antes,
    "Después de KNN": faltantes_despues
})

comparacion_nulos.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Comparación de valores faltantes")
plt.ylabel("Cantidad de valores NaN")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("comparacion_nulos.png", dpi=200)
plt.close()

# 9. VALIDACIÓN FINAL
print("\n=== VALIDACIÓN FINAL ===")

if df_knn2.isnull().any().any():
    print("Advertencia: todavía existen valores faltantes.")
else:
    print("Limpieza completada correctamente.")
    print("No existen valores faltantes.")

# 10. EXPORTACIÓN DE RESULTADOS
df_original.to_csv("dataset_original.csv", index=False)
df_knn2.to_csv("dataset_limpio.csv", index=False)
df_comparacion.to_csv(
    "comparacion_imputaciones.csv", index=False
)

print("\n=== ARCHIVOS GENERADOS ===")
print("dataset_original.csv")
print("dataset_limpio.csv")
print("comparacion_imputaciones.csv")
print("comparacion_boxplots.png")
print("comparacion_nulos.png")

print("\nPRÁCTICA 3 FINALIZADA CORRECTAMENTE")
