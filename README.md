# Caso de Estudio N°1 - Módulo 2

Aplicación interactiva desarrollada en Python con Streamlit para realizar un Análisis Exploratorio de Datos (EDA) del dataset `BankMarketing.csv`.

El objetivo no es construir modelos predictivos, sino explorar y describir el comportamiento de los clientes frente a una campaña de marketing bancario.

## Estructura de la aplicación

La aplicación principal se encuentra en `app.py` y está organizada en los siguientes módulos accesibles desde la barra lateral:

- **Home**: Presentación del proyecto, objetivos, descripción del dataset y tecnologías utilizadas.
- **Carga del dataset**: Carga del archivo `.csv` mediante `st.file_uploader`, validación básica, dimensiones y vista previa del dataset.
- **Análisis Exploratorio de Datos**: Núcleo del proyecto. Incluye al menos 10 ítems de análisis, organizados en pestañas (`st.tabs`) y columnas (`st.columns`), usando una clase de apoyo para el procesamiento del dataset.
- **Conclusiones**: Espacio para redactar conclusiones finales a partir del EDA.

Se utiliza Programación Orientada a Objetos a través de una clase que encapsula:

- Cálculo de estadísticas descriptivas
- Clasificación de variables numéricas y categóricas
- Resúmenes por la variable objetivo `y`
- Tablas cruzadas entre variables categóricas

## Funcionalidades principales (EDA)

Algunas de las funcionalidades incluidas:

- Información general del dataset (columnas, tipos, nulos, dimensiones).
- Clasificación de variables numéricas y categóricas.
- Estadísticas descriptivas con `pandas` y `numpy`.
- Análisis de valores faltantes y, en su caso, visualización de su porcentaje.
- Distribución de variables numéricas mediante histogramas (`matplotlib`/`seaborn`), con opción de colorear por la variable objetivo `y`.
- Análisis de variables categóricas con tablas de frecuencias y gráficos de barras.
- Análisis bivariado:
  - Numérico vs. objetivo (`y`)
  - Categórico vs. objetivo (`y`)
- Análisis dinámico en el que el usuario puede seleccionar variables y aplicar filtros (por ejemplo, rangos de edad y duración de llamada).
- Sección de hallazgos clave para resumir los principales insights.

## Capturas de la aplicación


## Requisitos

Las dependencias mínimas se encuentran en `requirements.txt`:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
```

## Cómo ejecutar la aplicación
1. Clonar el repositorio o descargar los archivos y descomprimir
2. Instalar dependencias (streamlit, la cual instalará otras más, matplotlib, seaborn) teniendo instalado Python.
3. Ejecutar en la terminal el siguiente comando: streamlit run app.py

## Enlaces

- Repositorio GitHub: (colocar enlace)
- Aplicación en Streamlit Cloud: (colocar enlace)