# TREESHARES INVESTMENT

Sistema de Analisis de Inversiones Usando Estructuras de Datos Jerarquicas y Grafos

Universidad del Rosario - Algoritmos y Estructuras de Datos

---

## Integrantes

- Carlos Gutierrez - Gerente de Proyecto
- Samuel Valderrama - Director de Pruebas
- David Pascagaza - Director de Diseno

---

## Descripcion

TreeShares Investment es un sistema de recomendacion de inversiones que analiza datos historicos de acciones para predecir cuales tienen mayor probabilidad de superar el benchmark del mercado.

El proyecto implementa tres estructuras de datos avanzadas desde cero, sin utilizar librerias de machine learning como sklearn.

---

## Estructuras de Datos Implementadas

### 1. Arbol de Decision

Se implemento el algoritmo de construccion de arboles de decision usando entropia y ganancia de informacion.

La entropia mide el desorden en los datos:

    H(S) = -sum( p(x) * log2(p(x)) )

La ganancia de informacion determina que caracteristica usar para dividir:

    IG(S,A) = H(S) - sum( (|Sv|/|S|) * H(Sv) )

El arbol se construye recursivamente, seleccionando en cada nodo la caracteristica que maximiza la ganancia de informacion. Se implementaron funciones para:
- Calcular entropia de un conjunto
- Calcular ganancia de informacion para cada division
- Encontrar el mejor punto de corte (umbral)
- Construir el arbol recursivamente
- Predecir recorriendo el arbol desde la raiz

### 2. Arbol Binario de Busqueda (BST)

Se implemento un BST para organizar las acciones por precio, permitiendo busquedas eficientes.

Complejidades:
- Insercion: O(log n)
- Busqueda: O(log n)
- Busqueda por rango: O(log n + k), donde k es el numero de resultados

Funcionalidades implementadas:
- Insercion recursiva manteniendo la propiedad del BST
- Busqueda por rango de precios
- Obtencion del minimo y maximo

### 3. Grafo de Correlaciones

Se implemento un grafo no dirigido ponderado usando listas de adyacencia.

- Vertices: cada accion (ticker) es un vertice
- Aristas: conectan acciones con correlacion alta en sus rendimientos
- Peso: valor de la correlacion entre los rendimientos historicos

Se implemento el algoritmo BFS (Breadth-First Search) para explorar acciones relacionadas, con complejidad O(V + E).

Funcionalidades:
- Encontrar acciones similares (alta correlacion)
- Encontrar acciones para diversificar (baja correlacion)
- Explorar relaciones de segundo grado con BFS

---

## Logica del Sistema

1. Se cargan los datos historicos del CSV (602,962 filas, 491 empresas)

2. Se preprocesa la informacion calculando para cada empresa:
   - Precio actual
   - Rendimiento total (5 anos)
   - Volatilidad (desviacion estandar de cambios diarios)
   - Indicadores financieros (RSI, ROE, P/E, margen EBITDA, deuda/EBITDA)

3. Se define el target: una accion "supera" si su rendimiento es mayor a la mediana

4. Se construye el BST insertando cada accion ordenada por precio

5. Se construye el grafo calculando correlaciones entre rendimientos diarios

6. Se entrena el arbol de decision con 80% de los datos

7. Se evalua el modelo con metricas: accuracy, precision, recall, F1

8. Se generan recomendaciones usando las predicciones del arbol

9. Se optimizan portafolios usando el grafo para diversificar

---

## Dataset

- Fuente: Kaggle - Yahoo Finance Dataset
- Archivo: stock_details_5_years.csv
- Registros: 602,962 filas
- Empresas: 491 (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, etc.)
- Periodo: 5 anos de datos historicos
- Columnas: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Company

---

## Ejecucion

Requisitos:
    pip install pandas numpy

Ejecutar:
    python main.py

---

## Funcionalidades del Menu

1. Cargar datos - Importa el CSV y construye las estructuras
2. Entrenar modelo - Construye el arbol de decision
3. Evaluar modelo - Muestra accuracy, precision, recall, F1
4. Buscar por precio - Busqueda en el BST por rango de precios
5. Correlaciones - Analisis con el grafo y BFS
6. Recomendaciones - Top 15 acciones predichas como ganadoras
7. Simular portafolio - Proyeccion con acciones elegidas por el usuario
8. Portafolio optimo - Seleccion automatica diversificada
9. Ver arbol - Visualizacion de las reglas de decision
10. Estadisticas - Metricas de las estructuras de datos

---

Universidad del Rosario - 2025
