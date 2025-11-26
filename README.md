# 🌳 TREESHARES INVESTMENT

## Sistema de Análisis de Inversiones Usando Estructuras de Datos Jerárquicas y Grafos

**Universidad del Rosario - Algoritmos y Estructuras de Datos**

---

## 👥 Integrantes

| Nombre | Rol |
|--------|-----|
| Carlos Gutiérrez | Gerente de Proyecto |
| Samuel Valderrama | Director de Pruebas |
| David Pascagaza | Director de Diseño |

---

## 📋 Descripción del Proyecto

TreeShares Investment es un sistema de recomendación de inversiones que implementa **tres estructuras de datos avanzadas desde cero** (sin usar librerías de ML como sklearn):

1. **Árbol de Decisión** - Para predicción de acciones ganadoras
2. **Árbol Binario de Búsqueda (BST)** - Para búsqueda eficiente por precios
3. **Grafo No Dirigido Ponderado** - Para análisis de correlaciones

---

## 🎯 Objetivos

### Objetivo General
Diseñar e implementar un algoritmo de Árbol de Decisión basado en estructuras de datos jerárquicas que recomiende activos financieros con alto potencial de superar el benchmark.

### Objetivos Específicos
- ✅ Desarrollar un Árbol de Decisión que prediga si una acción superará el S&P 500
- ✅ Construir un Grafo no dirigido ponderado para modelar correlaciones
- ✅ Implementar un BST para organizar acciones por precio
- ✅ Integrar las tres estructuras en un flujo coherente
- ✅ Procesar datos históricos de Yahoo Finance (+600,000 filas)
- ✅ Evaluar el modelo con métricas cuantitativas (accuracy, precision, recall, F1)
- ✅ Generar visualizaciones del Árbol de Decisión
- ✅ Proveer un prototipo funcional con simulación de portafolios

---

## 🔧 Estructuras de Datos Implementadas

### 1. Árbol de Decisión (Desde Cero)

```
Algoritmo: ID3/CART con Entropía y Ganancia de Información

Entropía: H(S) = -Σ p(x) · log₂(p(x))
Ganancia: IG(S,A) = H(S) - Σ (|Sᵥ|/|S|) · H(Sᵥ)
```

**Características:**
- Construcción recursiva del árbol
- Poda por profundidad máxima
- Cálculo de importancia de características
- Predicción por recorrido de nodos

### 2. Árbol Binario de Búsqueda (BST)

```
Complejidad:
- Inserción: O(log n)
- Búsqueda: O(log n)
- Búsqueda por rango: O(log n + k)
```

**Funcionalidades:**
- Organización de acciones por precio
- Búsqueda eficiente por rangos de precio
- Obtención de mínimo/máximo en O(log n)

### 3. Grafo de Correlaciones

```
Representación: Lista de Adyacencias
Algoritmo de búsqueda: BFS (Breadth-First Search)
Complejidad BFS: O(V + E)
```

**Funcionalidades:**
- Vértices: Acciones (tickers)
- Aristas: Correlaciones entre rendimientos
- Búsqueda de acciones similares
- Identificación de acciones para diversificación

---

## 📊 Dataset

- **Fuente:** Kaggle - Yahoo Finance Dataset
- **Archivo:** `stock_details_5_years.csv`
- **Filas:** 602,962
- **Empresas:** 491 (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, etc.)
- **Período:** 5 años de datos históricos
- **Columnas:** Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Company

---

## 🚀 Instalación y Uso

### Requisitos
```bash
pip install pandas numpy
```

### Ejecución
```bash
python main.py
```

### Menú Principal
```
============================================================
  MENU PRINCIPAL - TREESHARES INVESTMENT
============================================================

--- DATOS ---
1. Cargar datos desde CSV

--- MODELO ---
2. Entrenar Arbol de Decision
3. Evaluar modelo (Accuracy, Precision, Recall, F1)

--- ESTRUCTURAS DE DATOS ---
4. Buscar por precio (BST)
5. Analizar correlaciones (Grafo + BFS)

--- INVERSIONES ---
6. Ver recomendaciones TOP 15
7. Simular portafolio personalizado
8. Generar portafolio optimo diversificado

--- VISUALIZACION ---
9. Ver Arbol de Decision
10. Estadisticas generales

0. Salir
```

---

## 📈 Funcionalidades

| Función | Descripción |
|---------|-------------|
| **Cargar datos** | Importa 602,962 registros del CSV |
| **Entrenar modelo** | Construye el árbol de decisión |
| **Evaluar modelo** | Muestra Accuracy, Precision, Recall, F1 |
| **Buscar por precio** | Búsqueda O(log n) en el BST |
| **Correlaciones** | Análisis con BFS en el grafo |
| **Recomendaciones** | TOP 15 acciones predichas como ganadoras |
| **Simular portafolio** | Proyección de inversión con acciones elegidas |
| **Portafolio óptimo** | Selección automática diversificada |
| **Ver árbol** | Visualización de reglas de decisión |
| **Estadísticas** | Métricas de las estructuras |

---

## 📐 Métricas de Evaluación

El sistema calcula:
- **Accuracy**: Porcentaje total de predicciones correctas
- **Precision**: De las predicciones "COMPRAR", cuántas fueron correctas
- **Recall**: De las acciones ganadoras reales, cuántas detectó
- **F1-Score**: Media armónica de Precision y Recall

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    TREESHARES INVESTMENT                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Árbol de   │  │     BST      │  │    Grafo     │  │
│  │   Decisión   │  │   Precios    │  │ Correlaciones│  │
│  │              │  │              │  │              │  │
│  │ • Entropía   │  │ • Inserción  │  │ • Adyacencias│  │
│  │ • Ganancia   │  │ • Búsqueda   │  │ • BFS        │  │
│  │ • Predicción │  │ • Rango      │  │ • Similares  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │              MÓDULO DE PORTAFOLIOS                  ││
│  │  • Simulación  • Optimización  • Diversificación   ││
│  └─────────────────────────────────────────────────────┘│
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │                 INTERFAZ DE MENÚ                    ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Licencia

Proyecto académico - Universidad del Rosario 2025

---

## 📧 Contacto

Para dudas o sugerencias, contactar al equipo del proyecto.
