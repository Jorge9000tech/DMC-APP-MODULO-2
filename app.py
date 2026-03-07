import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Clase BankMarketingAnalyzer (POO)
class BankMarketingAnalyzer:
    def __init__(self, df):
        #Guardamos una copia para no modificar el dataset original
        self.df = df.copy()

    def obtener_columnas_numericas(self):
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def obtener_columnas_categoricas(self):
        return self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    def resumen_informacion(self):
        print(self.df.info())
        #df.info():
        datos_info = []
        for col in self.df.columns:
            no_nulos = self.df[col].notna().sum()
            nulos = self.df[col].isna().sum()
            tipo_dato = self.df[col].dtype
            datos_info.append(
                {
                    "Columna": col,
                    "Tipo de Dato": str(tipo_dato),
                    "No Nulos": no_nulos,
                    "Nulos": nulos,
                }
            )
        return pd.DataFrame(datos_info)

    def valores_faltantes(self):
        conteo_nulos = self.df.isna().sum()
        porcentaje_nulos = (conteo_nulos / len(self.df)) * 100
        resultados = pd.DataFrame(
            {
                "Columna": self.df.columns,
                "Nulos": conteo_nulos.values,
                "Porcentaje Nulos (%)": porcentaje_nulos.values,
            }
        )
        return resultados[resultados["Nulos"] > 0].sort_values("Porcentaje Nulos (%)", ascending=False)

    def estadisticas_descriptivas(self):
        columnas_num = self.obtener_columnas_numericas()
        return self.df[columnas_num].describe().T

    def conteo_categorias(self, columna):
        conteo = self.df[columna].value_counts(dropna=False)
        porcentaje = self.df[columna].value_counts(dropna=False, normalize=True) * 100
        return pd.DataFrame({"Conteo": conteo, "Porcentaje (%)": porcentaje})

    def agrupar_media_por_objetivo(self, columna_num, objetivo="y"):
        return (
            self.df.groupby(objetivo)[columna_num]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(columns={"count": "Conteo", "mean": "Media", "median": "Mediana"})
        )

    def tabla_cruzada(self, columna_cat, objetivo="y"):
        tabla = pd.crosstab(self.df[columna_cat], self.df[objetivo], margins=False)
        tabla_porcentaje = tabla.div(tabla.sum(axis=1), axis=0) * 100
        tabla_porcentaje = tabla_porcentaje.add_suffix(" (%)")
        return pd.concat([tabla, tabla_porcentaje], axis=1)

def main():
    st.title("Tarea Modulo 2 - Caso de estudio 1")
    menu_principal = ["Home", "Carga del dataset", "Análisis Exploratorio de Datos","Conclusiones"]
    menu_seleccion = st.sidebar.selectbox("Menu principal", menu_principal)

    if menu_seleccion == "Home":
        st.subheader("Home")
        st.write("""
        - **Título del proyecto:** Caso de estudio Bank Marketing 
        - **Descripción del objetivo:** Realizar un análisis del dataset del caso 1 y mostrar los resultados en una página en Streamlit
        - **Nombre completo del estudiante:** Jorge Ruiz Santillán
        - **Nombre del curso o módulo:** Especialización en Python for Analytics - Módulo 2
        - **Año:** 2026
        - **Descripción del dataaset**: El dataset pertenece a una entidad financiera con los datos de sus clientes, en la cual se busca analizar la efectividad de las campañas de marketing de la empresa
        - **Tecnologías usadas**: Streamlit, Pandas, Numpy, Matplotlib, Seaborn
        """)
    elif menu_seleccion == "Carga del dataset":
        st.subheader("Caso de estudio N°1 - Bank Marketing")
        #Cargar en dataset con un file uploader
        dataset = st.file_uploader("**Cargue el dataset aquí:**",type=['csv'])

        if dataset is not None:
            #Cargar el dataset como dataframe de pandas y guardarlo en la memoria de la página
            df = pd.read_csv(dataset, sep=";", encoding="utf-8")
            st.session_state["dataset"] = df
            st.success("¡Dataset cargado y guardado en memoria exitosamente!")
            st.write("##### Dataset: ")
            st.write(df)
            st.write("##### Vista previa: ")
            st.write(df.head())
            st.write("##### Dimensiones del dataset: ")
            st.write(df.shape)

    elif menu_seleccion == "Análisis Exploratorio de Datos":
        def grafico_histograma(df, columna, objetivo=None):
            fig, ax = plt.subplots(figsize=(6, 4))
            if objetivo and objetivo in df.columns:
                sns.histplot(data=df, x=columna, hue=objetivo, kde=False, ax=ax, multiple="stack")
            else:
                sns.histplot(data=df, x=columna, kde=False, ax=ax)
            ax.set_title(f"Distribución de {columna}")
            st.pyplot(fig)

        def grafico_de_barras(df, columna, objetivo=None):
            fig, ax = plt.subplots(figsize=(6, 4))
            if objetivo and objetivo in df.columns:
                sns.countplot(data=df, x=columna, hue=objetivo, ax=ax)
            else:
                sns.countplot(data=df, x=columna, ax=ax)
            plt.xticks(rotation=45, ha="right")
            ax.set_title(f"Distribución de {columna}")
            st.pyplot(fig)

        #Solo se ejecuta el análisis si se ha cargado el dataset previamente
        if "dataset" in st.session_state:
            df= st.session_state["dataset"]
            #Instancia de la clase BankMarketingAnalyzer para poder usar sus propiedades y funciones
            analyzer = BankMarketingAnalyzer(df)
            st.subheader("Módulo 3 - Análisis Exploratorio de Datos (EDA)")

            #Utilización de tabs para ver los items en 4 categorías utilizando columnas
            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "Información general",
                    "Distribuciones de variables",
                    "Análisis bivariado",
                    "Hallazgos clave",
                ]
            )
            with tab1:
                st.header("Ítem 1: Información general del dataset")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Resumen de columnas")
                    st.dataframe(analyzer.resumen_informacion())
                with col_b:
                    st.subheader("Dimensiones")
                    st.metric("Número de filas", f"{df.shape[0]:,}")
                    st.metric("Número de columnas", f"{df.shape[1]:,}")

                st.markdown("---")
                st.header("Ítem 2: Clasificación de variables")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Variables numéricas")
                    st.write(analyzer.obtener_columnas_numericas())
                    st.write(f"Total: {len(analyzer.obtener_columnas_numericas())}")
                with col2:
                    st.subheader("Variables categóricas")
                    st.write(analyzer.obtener_columnas_categoricas())
                    st.write(f"Total: {len(analyzer.obtener_columnas_categoricas())}")

                st.markdown("---")
                st.header("Ítem 3: Estadísticas descriptivas")
                st.dataframe(analyzer.estadisticas_descriptivas())

                st.markdown("---")
                st.header("Ítem 4: Análisis de valores faltantes")
                #Variable para realizar el análisis de valores faltantes
                vf = analyzer.valores_faltantes()
                if vf.empty:
                    st.success("No se encontraron valores faltantes en el dataset.")
                else:
                    st.dataframe(vf)
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=vf, x="columna", y="porcentaje_nulos", ax=ax)
                    plt.xticks(rotation=45, ha="right")
                    ax.set_ylabel("% de nulos")
                    st.pyplot(fig)
                    st.info(
                        "Revisa las variables con mayor porcentaje de valores faltantes "
                        "para decidir posibles estrategias de imputación o exclusión."
                    )
            with tab2:
                st.header("Ítem 5: Distribución de variables numéricas")
                num_col = st.selectbox(
                    "Selecciona una variable numérica:",
                    options=analyzer.obtener_columnas_numericas(),
                    key="num_hist",
                )
                variable_objetivo = st.checkbox(
                    "Colorear por variable objetivo (`y`)",
                    value=True,
                    key="num_hist_target",
                )
                if num_col:
                    grafico_histograma(df, num_col, "y" if variable_objetivo else None)

                st.markdown("---")
                st.header("Ítem 6: Análisis de variables categóricas")
                cat_col = st.selectbox(
                    "Selecciona una variable categórica:",
                    options=[c for c in analyzer.obtener_columnas_categoricas() if c != "y"],
                    key="cat_bar",
                )
                show_proportions = st.checkbox(
                    "Mostrar tabla de conteos y proporciones",
                    value=True,
                    key="cat_table",
                )
                if cat_col:
                    grafico_de_barras(df, cat_col, "y")
                    if show_proportions:
                        st.subheader("Tabla de frecuencias")
                        st.dataframe(analyzer.conteo_categorias(cat_col))

            with tab3:
                st.header("Ítem 7: Análisis bivariado (numérico vs objetivo)")
                st.write("Ejemplos sugeridos: `age` vs `y`, `duration` vs `y`.")
                num_bi = st.selectbox(
                    "Selecciona una variable numérica para comparar con `y`:",
                    options=analyzer.obtener_columnas_numericas(),
                    key="num_bi",
                )
                if num_bi:
                    st.subheader(f"Distribución de {num_bi} por `y`")
                    grafico_histograma(df, num_bi, "y")
                    st.subheader("Resumen estadístico por grupo de `y`")
                    st.dataframe(analyzer.agrupar_media_por_objetivo(num_bi))

                st.markdown("---")
                st.header("Ítem 8: Análisis bivariado (categórico vs categórico)")
                st.write("Ejemplos sugeridos: `education` vs `y`, `contact` vs `y`.")
                cat_bi = st.selectbox(
                    "Selecciona una variable categórica para comparar con `y`:",
                    options=[c for c in analyzer.obtener_columnas_categoricas() if c != "y"],
                    key="cat_bi",
                )
                if cat_bi:
                    st.subheader(f"Tabla cruzada {cat_bi} vs `y`")
                    st.dataframe(analyzer.tabla_cruzada(cat_bi))
                    st.subheader("Visualización")
                    grafico_de_barras(df, cat_bi, "y")

                st.markdown("---")
                st.header("Ítem 9: Análisis dinámico con parámetros seleccionados")
                st.write(
                    "Selecciona una o más variables numéricas para comparar su distribución "
                    "según la variable objetivo `y`."
                )
                selected_nums = st.multiselect(
                    "Variables numéricas:",
                    options=analyzer.obtener_columnas_numericas(),
                    default=["age", "duration"]
                    if "age" in analyzer.obtener_columnas_numericas() and "duration" in analyzer.obtener_columnas_numericas()
                    else analyzer.obtener_columnas_numericas()[:2],
                )

                if selected_nums:
                    for col_name in selected_nums:
                        st.subheader(f"Distribución de {col_name} por `y`")
                        grafico_histograma(df, col_name, "y")

                st.markdown("---")
                st.subheader("Filtros adicionales")
                with st.expander("Filtrar por rango de edad y duración de llamada"):
                    if "age" in analyzer.obtener_columnas_numericas():
                        min_age = int(df["age"].min())
                        max_age = int(df["age"].max())
                        age_range = st.slider(
                            "Rango de edad",
                            min_value=min_age,
                            max_value=max_age,
                            value=(min_age, max_age),
                        )
                    else:
                        age_range = None

                    if "duration" in analyzer.obtener_columnas_numericas():
                        min_dur = int(df["duration"].min())
                        max_dur = int(df["duration"].max())
                        dur_range = st.slider(
                            "Rango de duración de llamada (segundos)",
                            min_value=min_dur,
                            max_value=max_dur,
                            value=(min_dur, max_dur),
                        )
                    else:
                        dur_range = None

                    filtered_df = df.copy()
                    if age_range is not None:
                        filtered_df = filtered_df[
                            (filtered_df["age"] >= age_range[0])
                            & (filtered_df["age"] <= age_range[1])
                            ]
                    if dur_range is not None:
                        filtered_df = filtered_df[
                            (filtered_df["duration"] >= dur_range[0])
                            & (filtered_df["duration"] <= dur_range[1])
                            ]

                    st.write(
                        f"Filas filtradas: {len(filtered_df):,} "
                        f"(de un total de {len(df):,})"
                    )
                    show_filtered = st.checkbox(
                        "Mostrar datos filtrados (primeras 50 filas)",
                        value=False,
                    )
                    if show_filtered:
                        st.dataframe(filtered_df.head(50))

            with tab4:
                st.header("Ítem 10: Hallazgos clave")
                st.markdown(
                    """
                    #### Principales Hallazgos y Oportunidades

                    Tras explorar a fondo los datos de la última campaña para entender la reciente caída en la efectividad comercial, estos son los descubrimientos más relevantes que pueden ayudar a mejorar la estrategia de la empresa:

                    - **El perfil del cliente ideal:** Se identificó rangos de edad y niveles educativos (`education`) específicos que muestran una disposición mucho mayor a decir "sí". Enfocar los esfuerzos en estos segmentos podría recuperar rápidamente nuestras tasas de conversión.
                    - **El peso de la conversación:** Los datos revelan si mantener al cliente más tiempo en la línea (`duration`) realmente garantiza un cierre de venta exitoso, o si se debe entrenar a los ejecutivos para hacer llamadas más concisas y efectivas.
                    - **El canal correcto importa:** Al comparar las vías de comunicación (`contact`), quedó en evidencia qué canal genera el mayor retorno, lo que permite dejar de desperdiciar tiempo y recursos en medios menos efectivos.
                    - **El contexto económico manda:** Se identificaron ciertos valores externos a la empresa, como la variación del empleo (`emp.var.rate`) o el indicador Euribor (`euribor3m`), están impactando directamente en el bolsillo y la decisión del cliente. Entender esto es clave para ajustar las expectativas y el tono de la campaña.
                    """
                )
        else:
            st.write("### Dataset no encontrado, cargue el dataset primero")
    elif menu_seleccion=="Conclusiones":ejjeje
        st.subheader("Conclusiones")

if __name__ == '__main__':
    main()
