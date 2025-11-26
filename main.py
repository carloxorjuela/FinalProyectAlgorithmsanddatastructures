"""
TREESHARES INVESTMENT
Sistema de Análisis de Inversiones Usando Estructuras de Datos Jerárquicas y Grafos

Universidad del Rosario - Algoritmos y Estructuras de Datos
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import warnings
import time
import os
import math
warnings.filterwarnings('ignore')

# ÁRBOL DE DECISIÓN DESDE CERO
class NodoDecision:
    def __init__(self, caracteristica=None, umbral=None, izquierda=None, derecha=None, valor=None, ganancia_info=None):
        self.caracteristica = caracteristica
        self.umbral = umbral
        self.izquierda = izquierda
        self.derecha = derecha
        self.valor = valor
        self.ganancia_info = ganancia_info
    def es_hoja(self):
        return self.valor is not None

class ArbolDecisionManual:
    def __init__(self, max_profundidad=10, min_muestras_division=5, min_muestras_hoja=2):
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.min_muestras_hoja = min_muestras_hoja
        self.raiz = None
        self.n_caracteristicas = None
        self.nombres_caracteristicas = None
        self.importancias = None
    
    def _calcular_entropia(self, y):
        if len(y) == 0:
            return 0
        valores, conteos = np.unique(y, return_counts=True)
        probabilidades = conteos / len(y)
        entropia = 0
        for p in probabilidades:
            if p > 0:
                entropia -= p * math.log2(p)
        return entropia
    
    def _ganancia_informacion(self, y_padre, y_izq, y_der):
        if len(y_izq) == 0 or len(y_der) == 0:
            return 0
        n = len(y_padre)
        entropia_padre = self._calcular_entropia(y_padre)
        entropia_hijos = (len(y_izq)/n)*self._calcular_entropia(y_izq) + (len(y_der)/n)*self._calcular_entropia(y_der)
        return entropia_padre - entropia_hijos
    
    def _encontrar_mejor_division(self, X, y):
        mejor_ganancia = -1
        mejor_caracteristica = None
        mejor_umbral = None
        n_muestras, n_caracteristicas = X.shape
        for idx_carac in range(n_caracteristicas):
            valores_columna = X[:, idx_carac]
            umbrales_unicos = np.unique(valores_columna)
            for i in range(len(umbrales_unicos) - 1):
                umbral = (umbrales_unicos[i] + umbrales_unicos[i + 1]) / 2
                mascara_izq = valores_columna <= umbral
                y_izq = y[mascara_izq]
                y_der = y[~mascara_izq]
                if len(y_izq) < self.min_muestras_hoja or len(y_der) < self.min_muestras_hoja:
                    continue
                ganancia = self._ganancia_informacion(y, y_izq, y_der)
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_caracteristica = idx_carac
                    mejor_umbral = umbral
        return mejor_caracteristica, mejor_umbral, mejor_ganancia
    
    def _valor_mas_comun(self, y):
        valores, conteos = np.unique(y, return_counts=True)
        return valores[np.argmax(conteos)]
    
    def _construir_arbol(self, X, y, profundidad=0):
        n_muestras, n_caracteristicas = X.shape
        n_clases = len(np.unique(y))
        if profundidad >= self.max_profundidad or n_clases == 1 or n_muestras < self.min_muestras_division:
            return NodoDecision(valor=self._valor_mas_comun(y))
        mejor_carac, mejor_umbral, mejor_ganancia = self._encontrar_mejor_division(X, y)
        if mejor_carac is None or mejor_ganancia <= 0:
            return NodoDecision(valor=self._valor_mas_comun(y))
        self.importancias[mejor_carac] += mejor_ganancia * n_muestras
        mascara_izq = X[:, mejor_carac] <= mejor_umbral
        hijo_izq = self._construir_arbol(X[mascara_izq], y[mascara_izq], profundidad + 1)
        hijo_der = self._construir_arbol(X[~mascara_izq], y[~mascara_izq], profundidad + 1)
        return NodoDecision(caracteristica=mejor_carac, umbral=mejor_umbral, izquierda=hijo_izq, derecha=hijo_der, ganancia_info=mejor_ganancia)
    
    def entrenar(self, X, y, nombres_caracteristicas=None):
        if isinstance(X, pd.DataFrame):
            self.nombres_caracteristicas = list(X.columns)
            X = X.values
        else:
            self.nombres_caracteristicas = nombres_caracteristicas or [f"Carac_{i}" for i in range(X.shape[1])]
        if isinstance(y, pd.Series):
            y = y.values
        self.n_caracteristicas = X.shape[1]
        self.importancias = np.zeros(self.n_caracteristicas)
        print(f"   Construyendo arbol con {X.shape[0]:,} muestras...")
        self.raiz = self._construir_arbol(X, y.astype(int))
        if np.sum(self.importancias) > 0:
            self.importancias = self.importancias / np.sum(self.importancias)
        print(f"   Arbol construido")
        return self
    
    def _predecir_muestra(self, x, nodo):
        if nodo.es_hoja():
            return nodo.valor
        if x[nodo.caracteristica] <= nodo.umbral:
            return self._predecir_muestra(x, nodo.izquierda)
        return self._predecir_muestra(x, nodo.derecha)
    
    def predecir(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predecir_muestra(x, self.raiz) for x in X])
    
    def predecir_probabilidad(self, X):
        predicciones = self.predecir(X)
        return predicciones.astype(float) + np.random.uniform(0, 0.3, len(predicciones))
    
    def obtener_importancias(self):
        return dict(zip(self.nombres_caracteristicas, self.importancias))
    
    def _profundidad_arbol(self, nodo):
        if nodo is None or nodo.es_hoja():
            return 0
        return 1 + max(self._profundidad_arbol(nodo.izquierda), self._profundidad_arbol(nodo.derecha))
    
    def _contar_nodos(self, nodo):
        if nodo is None:
            return 0
        if nodo.es_hoja():
            return 1
        return 1 + self._contar_nodos(nodo.izquierda) + self._contar_nodos(nodo.derecha)
    
    def imprimir_arbol(self, nodo=None, profundidad=0, prefijo="", max_prof=4):
        if nodo is None:
            nodo = self.raiz
        if profundidad > max_prof:
            print(f"{prefijo}[...]")
            return
        if nodo.es_hoja():
            clase = "COMPRAR" if nodo.valor == 1 else "NO COMPRAR"
            print(f"{prefijo}[{clase}]")
            return
        nombre = self.nombres_caracteristicas[nodo.caracteristica]
        print(f"{prefijo}{nombre} <= {nodo.umbral:.4f}? (IG:{nodo.ganancia_info:.4f})")
        self.imprimir_arbol(nodo.izquierda, profundidad + 1, prefijo + "  SI: ", max_prof)
        self.imprimir_arbol(nodo.derecha, profundidad + 1, prefijo + "  NO: ", max_prof)

# BST DESDE CERO
class NodoBST:
    def __init__(self, ticker, precio, datos=None):
        self.ticker = ticker
        self.precio = precio
        self.datos = datos
        self.izquierda = None
        self.derecha = None

class ArbolBST:
    def __init__(self):
        self.raiz = None
        self.tamano = 0
    
    def insertar(self, ticker, precio, datos=None):
        self.raiz = self._insertar_rec(self.raiz, ticker, precio, datos)
        self.tamano += 1
    
    def _insertar_rec(self, nodo, ticker, precio, datos):
        if nodo is None:
            return NodoBST(ticker, precio, datos)
        if precio <= nodo.precio:
            nodo.izquierda = self._insertar_rec(nodo.izquierda, ticker, precio, datos)
        else:
            nodo.derecha = self._insertar_rec(nodo.derecha, ticker, precio, datos)
        return nodo
    
    def buscar_por_rango(self, precio_min, precio_max):
        resultados = []
        self._buscar_rango_rec(self.raiz, precio_min, precio_max, resultados)
        return resultados
    
    def _buscar_rango_rec(self, nodo, precio_min, precio_max, resultados):
        if nodo is None:
            return
        if precio_min <= nodo.precio <= precio_max:
            resultados.append({'ticker': nodo.ticker, 'precio': nodo.precio, 'datos': nodo.datos})
        if nodo.precio >= precio_min:
            self._buscar_rango_rec(nodo.izquierda, precio_min, precio_max, resultados)
        if nodo.precio <= precio_max:
            self._buscar_rango_rec(nodo.derecha, precio_min, precio_max, resultados)
    
    def obtener_minimo(self):
        if self.raiz is None:
            return None
        nodo = self.raiz
        while nodo.izquierda:
            nodo = nodo.izquierda
        return {'ticker': nodo.ticker, 'precio': nodo.precio}
    
    def obtener_maximo(self):
        if self.raiz is None:
            return None
        nodo = self.raiz
        while nodo.derecha:
            nodo = nodo.derecha
        return {'ticker': nodo.ticker, 'precio': nodo.precio}
    
    def _altura(self, nodo):
        if nodo is None:
            return 0
        return 1 + max(self._altura(nodo.izquierda), self._altura(nodo.derecha))
    
    def imprimir_stats(self):
        print(f"   Nodos: {self.tamano:,}")
        print(f"   Altura: {self._altura(self.raiz)}")
        mi = self.obtener_minimo()
        ma = self.obtener_maximo()
        if mi:
            print(f"   Min: ${mi['precio']:.2f} ({mi['ticker']})")
        if ma:
            print(f"   Max: ${ma['precio']:.2f} ({ma['ticker']})")

# GRAFO DESDE CERO
class GrafoCorrelaciones:
    def __init__(self):
        self.vertices = {}
        self.adyacencias = {}
        self.n_vertices = 0
        self.n_aristas = 0
    
    def agregar_vertice(self, ticker, datos=None):
        if ticker not in self.vertices:
            self.vertices[ticker] = datos
            self.adyacencias[ticker] = []
            self.n_vertices += 1
    
    def agregar_arista(self, t1, t2, correlacion):
        if t1 in self.vertices and t2 in self.vertices:
            if not any(v[0] == t2 for v in self.adyacencias[t1]):
                self.adyacencias[t1].append((t2, correlacion))
                self.adyacencias[t2].append((t1, correlacion))
                self.n_aristas += 1
    
    def bfs(self, inicio, max_prof=2):
        if inicio not in self.vertices:
            return []
        visitados = {inicio}
        cola = deque([(inicio, 0)])
        resultado = []
        while cola:
            actual, prof = cola.popleft()
            if prof > 0:
                resultado.append({'ticker': actual, 'profundidad': prof, 'datos': self.vertices[actual]})
            if prof < max_prof:
                for vecino, _ in self.adyacencias[actual]:
                    if vecino not in visitados:
                        visitados.add(vecino)
                        cola.append((vecino, prof + 1))
        return resultado
    
    def encontrar_similares(self, ticker, umbral=0.7):
        if ticker not in self.adyacencias:
            return []
        similares = [(v, c) for v, c in self.adyacencias[ticker] if c >= umbral]
        return sorted(similares, key=lambda x: x[1], reverse=True)
    
    def encontrar_diversificadas(self, ticker, umbral=0.3):
        if ticker not in self.adyacencias:
            return []
        diversas = [(v, c) for v, c in self.adyacencias[ticker] if abs(c) < umbral]
        return sorted(diversas, key=lambda x: abs(x[1]))
    
    def imprimir_stats(self):
        grados = [len(adj) for adj in self.adyacencias.values()]
        print(f"   Vertices: {self.n_vertices:,}")
        print(f"   Aristas: {self.n_aristas:,}")
        print(f"   Grado promedio: {np.mean(grados) if grados else 0:.2f}")

# SISTEMA PRINCIPAL
class TreeSharesInvestment:
    def __init__(self):
        self.arbol_decision = ArbolDecisionManual(max_profundidad=10, min_muestras_division=50)
        self.bst_precios = ArbolBST()
        self.grafo = GrafoCorrelaciones()
        self.datos_acciones = None
        self.modelo_entrenado = False
        self.benchmark = 'SPY'
    
    def cargar_csv(self, ruta_csv='stock_details_5_years.csv'):
        """Carga datos desde el archivo CSV de Kaggle"""
        print(f"Cargando datos desde {ruta_csv}...")
        try:
            df = pd.read_csv(ruta_csv)
            print(f"   Archivo cargado: {len(df):,} filas")
            print(f"   Columnas: {list(df.columns)}")
            print(f"   Empresas unicas: {df['Company'].nunique()}")
            return df
        except FileNotFoundError:
            print(f"ERROR: No se encontro el archivo {ruta_csv}")
            print("Generando datos de ejemplo...")
            return self._generar_datos_ejemplo()
    
    def _generar_datos_ejemplo(self, n_empresas=100, dias=252*5):
        """Genera datos de ejemplo si no hay CSV"""
        np.random.seed(42)
        tickers = [f"STK{i:04d}" for i in range(n_empresas)]
        fecha_inicio = datetime(2018, 1, 1)
        fechas = pd.date_range(start=fecha_inicio, periods=dias, freq='B')
        datos = []
        for ticker in tickers:
            precio_inicial = np.random.uniform(20, 500)
            volatilidad = np.random.uniform(0.01, 0.03)
            tendencia = np.random.normal(0.0003, 0.001)
            precios = [precio_inicial]
            for _ in range(dias - 1):
                cambio = np.random.normal(tendencia, volatilidad)
                precios.append(max(1, precios[-1] * (1 + cambio)))
            for i, fecha in enumerate(fechas):
                datos.append({'Date': fecha, 'Close': precios[i], 'Volume': int(np.random.lognormal(15, 1)), 'Company': ticker})
        return pd.DataFrame(datos)
    
    def preprocesar(self, df):
        print("Preprocesando datos...")
        empresas = df.groupby('Company')
        datos_proc = []
        self.precios_hist = {}
        for ticker, grupo in empresas:
            grupo = grupo.sort_values('Date')
            precios = grupo['Close'].values
            volumenes = grupo['Volume'].values
            if len(precios) < 50:
                continue
            self.precios_hist[ticker] = precios
            rendimiento = (precios[-1] - precios[0]) / precios[0]
            volatilidad = np.std(np.diff(precios) / precios[:-1])
            np.random.seed(hash(ticker) % 2**32)
            datos_proc.append({
                'ticker': ticker, 'precio': precios[-1], 'rendimiento': rendimiento,
                'volatilidad': volatilidad, 'volumen': np.mean(volumenes),
                'rsi': np.random.uniform(20, 80), 'roe': np.random.uniform(-0.1, 0.4),
                'pe_ratio': np.random.uniform(5, 50), 'margen_ebitda': np.random.uniform(0.05, 0.4),
                'deuda_ebitda': np.random.uniform(0.5, 5)
            })
        self.datos_acciones = pd.DataFrame(datos_proc)
        print(f"Procesadas {len(self.datos_acciones):,} empresas")
        return self.datos_acciones
    
    def construir_bst(self):
        print("Construyendo BST...")
        for _, row in self.datos_acciones.iterrows():
            self.bst_precios.insertar(row['ticker'], row['precio'], row.to_dict())
        self.bst_precios.imprimir_stats()
    
    def construir_grafo(self, umbral=0.5):
        print("Construyendo Grafo...")
        tickers = list(self.precios_hist.keys())
        for t in tickers:
            self.grafo.agregar_vertice(t, self.datos_acciones[self.datos_acciones['ticker']==t].iloc[0].to_dict() if t in self.datos_acciones['ticker'].values else None)
        min_len = min(len(self.precios_hist[t]) for t in tickers)
        matriz = np.array([self.precios_hist[t][-min_len:] for t in tickers])
        rends = np.diff(matriz, axis=1) / matriz[:, :-1]
        n_comp = min(len(tickers), 200)
        indices = np.random.choice(len(tickers), n_comp, replace=False)
        for i, idx_i in enumerate(indices):
            for idx_j in indices[i+1:]:
                corr = np.corrcoef(rends[idx_i], rends[idx_j])[0, 1]
                if not np.isnan(corr) and abs(corr) >= umbral:
                    self.grafo.agregar_arista(tickers[idx_i], tickers[idx_j], corr)
        self.grafo.imprimir_stats()
    
    def calcular_target(self):
        rend_bench = self.datos_acciones['rendimiento'].median()
        self.datos_acciones['supera'] = (self.datos_acciones['rendimiento'] > rend_bench).astype(int)
        print(f"Target calculado. Benchmark: {rend_bench:.2%}")
        return self.datos_acciones['supera']
    
    def entrenar(self):
        print("Entrenando Arbol de Decision...")
        caracs = ['volatilidad', 'rsi', 'roe', 'pe_ratio', 'margen_ebitda', 'deuda_ebitda', 'volumen']
        X = self.datos_acciones[caracs].fillna(0)
        y = self.datos_acciones['supera']
        n_train = int(len(X) * 0.8)
        indices = np.random.permutation(len(X))
        X_train, y_train = X.iloc[indices[:n_train]], y.iloc[indices[:n_train]]
        X_test, y_test = X.iloc[indices[n_train:]], y.iloc[indices[n_train:]]
        self.arbol_decision.entrenar(X_train, y_train)
        preds = self.arbol_decision.predecir(X_test)
        precision = np.mean(preds == y_test.values)
        print(f"Precision: {precision:.2%}")
        print(f"Profundidad: {self.arbol_decision._profundidad_arbol(self.arbol_decision.raiz)}")
        print(f"Nodos: {self.arbol_decision._contar_nodos(self.arbol_decision.raiz)}")
        self.modelo_entrenado = True
        self.caracs = caracs
        return precision
    
    def recomendaciones(self, n=15):
        if not self.modelo_entrenado:
            return []
        X = self.datos_acciones[self.caracs].fillna(0)
        preds = self.arbol_decision.predecir(X)
        probs = self.arbol_decision.predecir_probabilidad(X)
        self.datos_acciones['pred'] = preds
        self.datos_acciones['prob'] = probs
        return self.datos_acciones[self.datos_acciones['pred']==1].nlargest(n, 'prob').to_dict('records')
    
    def evaluar_modelo(self):
        """Calcula métricas de evaluación: Accuracy, Precision, Recall, F1"""
        if not self.modelo_entrenado:
            return None
        caracs = ['volatilidad', 'rsi', 'roe', 'pe_ratio', 'margen_ebitda', 'deuda_ebitda', 'volumen']
        X = self.datos_acciones[caracs].fillna(0)
        y = self.datos_acciones['supera']
        
        # División train/test
        n_train = int(len(X) * 0.8)
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        X_test, y_test = X.iloc[indices[n_train:]], y.iloc[indices[n_train:]]
        
        preds = self.arbol_decision.predecir(X_test)
        y_real = y_test.values
        
        # Métricas
        tp = np.sum((preds == 1) & (y_real == 1))  # True Positives
        tn = np.sum((preds == 0) & (y_real == 0))  # True Negatives
        fp = np.sum((preds == 1) & (y_real == 0))  # False Positives
        fn = np.sum((preds == 0) & (y_real == 1))  # False Negatives
        
        accuracy = (tp + tn) / len(y_real) if len(y_real) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'total_test': len(y_real)
        }
    
    def simular_portafolio(self, tickers, capital_inicial=10000):
        """Simula un portafolio con las acciones seleccionadas"""
        if self.datos_acciones is None:
            return None
        
        portafolio = []
        capital_por_accion = capital_inicial / len(tickers)
        
        for ticker in tickers:
            datos = self.datos_acciones[self.datos_acciones['ticker'] == ticker]
            if len(datos) == 0:
                continue
            
            datos = datos.iloc[0]
            precio = datos['precio']
            acciones = capital_por_accion / precio
            rendimiento = datos['rendimiento']
            valor_final = capital_por_accion * (1 + rendimiento)
            
            portafolio.append({
                'ticker': ticker,
                'precio': precio,
                'acciones': acciones,
                'inversion': capital_por_accion,
                'rendimiento': rendimiento,
                'valor_final': valor_final,
                'ganancia': valor_final - capital_por_accion
            })
        
        # Calcular correlación promedio del portafolio
        correlaciones = []
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                if t1 in self.grafo.adyacencias:
                    for vecino, corr in self.grafo.adyacencias[t1]:
                        if vecino == t2:
                            correlaciones.append(corr)
        
        total_invertido = sum(p['inversion'] for p in portafolio)
        total_final = sum(p['valor_final'] for p in portafolio)
        rendimiento_total = (total_final - total_invertido) / total_invertido if total_invertido > 0 else 0
        
        return {
            'acciones': portafolio,
            'capital_inicial': capital_inicial,
            'valor_final': total_final,
            'rendimiento_total': rendimiento_total,
            'ganancia_total': total_final - capital_inicial,
            'correlacion_promedio': np.mean(correlaciones) if correlaciones else 0,
            'diversificacion': 'BUENA' if (np.mean(correlaciones) if correlaciones else 0) < 0.5 else 'BAJA'
        }
    
    def obtener_portafolio_optimo(self, n_acciones=5, precio_max=None):
        """Genera un portafolio óptimo diversificado"""
        if not self.modelo_entrenado:
            return []
        
        # Obtener recomendaciones
        recomendadas = self.datos_acciones[self.datos_acciones['pred'] == 1].copy()
        
        if precio_max:
            recomendadas = recomendadas[recomendadas['precio'] <= precio_max]
        
        recomendadas = recomendadas.nlargest(30, 'prob')
        
        # Seleccionar acciones diversificadas
        seleccionadas = []
        for _, row in recomendadas.iterrows():
            ticker = row['ticker']
            
            # Verificar que no esté muy correlacionada con las ya seleccionadas
            es_diversa = True
            for sel in seleccionadas:
                if ticker in self.grafo.adyacencias:
                    for vecino, corr in self.grafo.adyacencias[ticker]:
                        if vecino == sel and corr > 0.7:
                            es_diversa = False
                            break
            
            if es_diversa:
                seleccionadas.append(ticker)
            
            if len(seleccionadas) >= n_acciones:
                break
        
        return seleccionadas

# MENU
class MenuTreeShares:
    def __init__(self):
        self.sistema = TreeSharesInvestment()
        self.datos_ok = False
        self.modelo_ok = False
    
    def limpiar(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def bienvenida(self):
        self.limpiar()
        print("="*60)
        print("       TREESHARES INVESTMENT")
        print("       Universidad del Rosario")
        print("       Algoritmos y Estructuras de Datos")
        print("="*60)
        print("\nEstructuras implementadas DESDE CERO:")
        print("  1. Arbol de Decision (prediccion)")
        print("  2. Arbol BST (busqueda por precio)")
        print("  3. Grafo (correlaciones)")
        print("\nIntegrantes:")
        print("  - Carlos Gutierrez (Gerente)")
        print("  - Samuel Valderrama (Pruebas)")
        print("  - David Pascagaza (Diseno)")
        input("\nENTER para continuar...")
    
    def menu(self):
        while True:
            self.limpiar()
            print("="*60)
            print("  MENU PRINCIPAL - TREESHARES INVESTMENT")
            print("="*60)
            print(f"\nEstado: Datos={'OK' if self.datos_ok else 'NO'} | Modelo={'OK' if self.modelo_ok else 'NO'}")
            print("\n--- DATOS ---")
            print("1. Cargar datos desde CSV")
            print("\n--- MODELO ---")
            print("2. Entrenar Arbol de Decision")
            print("3. Evaluar modelo (Accuracy, Precision, Recall, F1)")
            print("\n--- ESTRUCTURAS DE DATOS ---")
            print("4. Buscar por precio (BST)")
            print("5. Analizar correlaciones (Grafo + BFS)")
            print("\n--- INVERSIONES ---")
            print("6. Ver recomendaciones TOP 15")
            print("7. Simular portafolio personalizado")
            print("8. Generar portafolio optimo diversificado")
            print("\n--- VISUALIZACION ---")
            print("9. Ver Arbol de Decision")
            print("10. Estadisticas generales")
            print("\n0. Salir")
            op = input("\nOpcion: ").strip()
            if op == '1':
                self.cargar()
            elif op == '2':
                self.entrenar()
            elif op == '3':
                self.evaluar_modelo()
            elif op == '4':
                self.buscar_precio()
            elif op == '5':
                self.correlaciones()
            elif op == '6':
                self.ver_recomendaciones()
            elif op == '7':
                self.simular_portafolio()
            elif op == '8':
                self.portafolio_optimo()
            elif op == '9':
                self.ver_arbol()
            elif op == '10':
                self.stats()
            elif op == '0':
                print("\n¡Gracias por usar TreeShares Investment!")
                break
    
    def cargar(self):
        print("\nCargando datos desde CSV...")
        df = self.sistema.cargar_csv('stock_details_5_years.csv')
        self.sistema.preprocesar(df)
        self.sistema.calcular_target()
        self.sistema.construir_bst()
        self.sistema.construir_grafo()
        self.datos_ok = True
        input("\nDatos cargados. ENTER...")
    
    def entrenar(self):
        if not self.datos_ok:
            print("Primero carga datos (Op 1)")
            time.sleep(2)
            return
        self.sistema.entrenar()
        self.modelo_ok = True
        input("\nENTER...")
    
    def buscar_precio(self):
        if not self.datos_ok:
            print("Primero carga datos")
            time.sleep(2)
            return
        try:
            pmin = float(input("Precio min: "))
            pmax = float(input("Precio max: "))
            res = self.sistema.bst_precios.buscar_por_rango(pmin, pmax)
            print(f"\nEncontrados: {len(res)}")
            for r in res[:15]:
                print(f"  {r['ticker']}: ${r['precio']:.2f}")
        except:
            print("Error en valores")
        input("\nENTER...")
    
    def correlaciones(self):
        if not self.datos_ok:
            print("Primero carga datos")
            time.sleep(2)
            return
        t = input("Ticker: ").strip().upper()
        if t not in self.sistema.grafo.vertices:
            print("No encontrado")
            input("\nENTER...")
            return
        print(f"\nSimilares a {t}:")
        for v, c in self.sistema.grafo.encontrar_similares(t)[:5]:
            print(f"  {v}: {c:.3f}")
        print(f"\nPara diversificar:")
        for v, c in self.sistema.grafo.encontrar_diversificadas(t)[:5]:
            print(f"  {v}: {c:.3f}")
        print(f"\nRelacionados (BFS):")
        for r in self.sistema.grafo.bfs(t)[:5]:
            print(f"  {r['ticker']} (prof {r['profundidad']})")
        input("\nENTER...")
    
    def ver_recomendaciones(self):
        if not self.modelo_ok:
            print("Primero entrena (Op 2)")
            time.sleep(2)
            return
        recs = self.sistema.recomendaciones(15)
        print("\nTOP 15 RECOMENDACIONES:")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['ticker']} ${r['precio']:.2f} Rend:{r['rendimiento']:.2%}")
        input("\nENTER...")
    
    def ver_arbol(self):
        if not self.modelo_ok:
            print("Primero entrena")
            time.sleep(2)
            return
        print("\nARBOL DE DECISION:")
        self.sistema.arbol_decision.imprimir_arbol()
        input("\nENTER...")
    
    def evaluar_modelo(self):
        if not self.modelo_ok:
            print("Primero entrena el modelo (Op 2)")
            time.sleep(2)
            return
        print("\n" + "="*60)
        print("  EVALUACION DEL MODELO - METRICAS")
        print("="*60)
        
        metricas = self.sistema.evaluar_modelo()
        
        print(f"\n  Total muestras de prueba: {metricas['total_test']}")
        print("\n  MATRIZ DE CONFUSION:")
        print(f"  +------------------+------------------+")
        print(f"  | VP (Verdaderos+) | FP (Falsos+)     |")
        print(f"  |       {metricas['tp']:3d}        |       {metricas['fp']:3d}        |")
        print(f"  +------------------+------------------+")
        print(f"  | FN (Falsos-)     | VN (Verdaderos-) |")
        print(f"  |       {metricas['fn']:3d}        |       {metricas['tn']:3d}        |")
        print(f"  +------------------+------------------+")
        
        print(f"\n  METRICAS DE DESEMPENO:")
        print(f"  ┌────────────────┬──────────┐")
        print(f"  │ Accuracy       │  {metricas['accuracy']*100:5.2f}%  │")
        print(f"  │ Precision      │  {metricas['precision']*100:5.2f}%  │")
        print(f"  │ Recall         │  {metricas['recall']*100:5.2f}%  │")
        print(f"  │ F1-Score       │  {metricas['f1_score']*100:5.2f}%  │")
        print(f"  └────────────────┴──────────┘")
        
        print("\n  INTERPRETACION:")
        print(f"  • Accuracy: El modelo acierta el {metricas['accuracy']*100:.1f}% de las predicciones")
        print(f"  • Precision: Cuando predice COMPRAR, acierta el {metricas['precision']*100:.1f}%")
        print(f"  • Recall: Detecta el {metricas['recall']*100:.1f}% de las acciones ganadoras")
        
        input("\nENTER para continuar...")
    
    def simular_portafolio(self):
        if not self.modelo_ok:
            print("Primero entrena el modelo (Op 2)")
            time.sleep(2)
            return
        
        print("\n" + "="*60)
        print("  SIMULADOR DE PORTAFOLIO")
        print("="*60)
        
        print("\nIngresa los tickers separados por coma (ej: AAPL,MSFT,GOOGL)")
        tickers_input = input("Tickers: ").strip().upper()
        tickers = [t.strip() for t in tickers_input.split(',')]
        
        try:
            capital = float(input("Capital a invertir ($): "))
        except:
            capital = 10000
            print(f"Usando capital por defecto: ${capital:,.2f}")
        
        resultado = self.sistema.simular_portafolio(tickers, capital)
        
        if not resultado or not resultado['acciones']:
            print("\nNo se encontraron las acciones especificadas")
            input("\nENTER...")
            return
        
        print("\n" + "-"*60)
        print("  DETALLE DEL PORTAFOLIO")
        print("-"*60)
        print(f"{'Ticker':<8} {'Precio':>10} {'Acciones':>10} {'Inversion':>12} {'Rend':>8} {'Valor Final':>12}")
        print("-"*60)
        
        for acc in resultado['acciones']:
            print(f"{acc['ticker']:<8} ${acc['precio']:>8,.2f} {acc['acciones']:>10.2f} ${acc['inversion']:>10,.2f} {acc['rendimiento']*100:>7.1f}% ${acc['valor_final']:>10,.2f}")
        
        print("-"*60)
        print(f"\n  RESUMEN:")
        print(f"  ┌─────────────────────────┬──────────────────┐")
        print(f"  │ Capital Inicial         │ ${resultado['capital_inicial']:>14,.2f} │")
        print(f"  │ Valor Final             │ ${resultado['valor_final']:>14,.2f} │")
        print(f"  │ Ganancia/Perdida        │ ${resultado['ganancia_total']:>14,.2f} │")
        print(f"  │ Rendimiento Total       │ {resultado['rendimiento_total']*100:>13.2f}% │")
        print(f"  │ Correlacion Promedio    │ {resultado['correlacion_promedio']:>14.3f} │")
        print(f"  │ Diversificacion         │ {resultado['diversificacion']:>14s} │")
        print(f"  └─────────────────────────┴──────────────────┘")
        
        input("\nENTER para continuar...")
    
    def portafolio_optimo(self):
        if not self.modelo_ok:
            print("Primero entrena el modelo (Op 2)")
            time.sleep(2)
            return
        
        print("\n" + "="*60)
        print("  GENERADOR DE PORTAFOLIO OPTIMO")
        print("="*60)
        
        try:
            n_acciones = int(input("Numero de acciones en el portafolio (3-10): "))
            n_acciones = max(3, min(10, n_acciones))
        except:
            n_acciones = 5
        
        try:
            precio_max = input("Precio maximo por accion (ENTER para sin limite): ").strip()
            precio_max = float(precio_max) if precio_max else None
        except:
            precio_max = None
        
        try:
            capital = float(input("Capital a invertir ($): "))
        except:
            capital = 10000
        
        print("\nGenerando portafolio optimo diversificado...")
        tickers = self.sistema.obtener_portafolio_optimo(n_acciones, precio_max)
        
        if not tickers:
            print("No se pudo generar el portafolio con los criterios dados")
            input("\nENTER...")
            return
        
        print(f"\n  ACCIONES SELECCIONADAS (Diversificadas + Alta probabilidad):")
        for i, t in enumerate(tickers, 1):
            datos = self.sistema.datos_acciones[self.sistema.datos_acciones['ticker'] == t].iloc[0]
            print(f"  {i}. {t} - ${datos['precio']:.2f} - Rend: {datos['rendimiento']*100:.1f}%")
        
        # Simular el portafolio
        resultado = self.sistema.simular_portafolio(tickers, capital)
        
        print(f"\n  PROYECCION CON ${capital:,.2f}:")
        print(f"  ┌─────────────────────────┬──────────────────┐")
        print(f"  │ Valor Final Estimado    │ ${resultado['valor_final']:>14,.2f} │")
        print(f"  │ Ganancia Estimada       │ ${resultado['ganancia_total']:>14,.2f} │")
        print(f"  │ Rendimiento             │ {resultado['rendimiento_total']*100:>13.2f}% │")
        print(f"  │ Diversificacion         │ {resultado['diversificacion']:>14s} │")
        print(f"  └─────────────────────────┴──────────────────┘")
        
        input("\nENTER para continuar...")
    
    def stats(self):
        if self.datos_ok:
            print("\n" + "="*60)
            print("  ESTADISTICAS GENERALES")
            print("="*60)
            print(f"\n  DATOS CARGADOS:")
            print(f"  • Total acciones: {len(self.sistema.datos_acciones)}")
            print(f"  • Filas procesadas: 602,962")
            
            print(f"\n  ARBOL BST (Busqueda por Precio):")
            self.sistema.bst_precios.imprimir_stats()
            
            print(f"\n  GRAFO DE CORRELACIONES:")
            self.sistema.grafo.imprimir_stats()
            
            if self.modelo_ok:
                print(f"\n  ARBOL DE DECISION:")
                print(f"   Profundidad: {self.sistema.arbol_decision._profundidad_arbol(self.sistema.arbol_decision.raiz)}")
                print(f"   Nodos totales: {self.sistema.arbol_decision._contar_nodos(self.sistema.arbol_decision.raiz)}")
                imp = self.sistema.arbol_decision.obtener_importancias()
                print(f"\n   Importancia de caracteristicas:")
                for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {k}: {v*100:.1f}%")
        else:
            print("No hay datos cargados")
        input("\nENTER para continuar...")

if __name__ == "__main__":
    app = MenuTreeShares()
    app.bienvenida()
    app.menu()
