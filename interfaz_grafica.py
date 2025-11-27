# -*- coding: utf-8 -*-
"""
TREESHARES INVESTMENT - INTERFAZ GRAFICA
Universidad del Rosario - Algoritmos y Estructuras de Datos
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import math
import os

# =====================================================================
# ESTRUCTURAS DE DATOS (Copiadas de main.py)
# =====================================================================

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
            if len(umbrales_unicos) > 20:
                umbrales_unicos = np.percentile(valores_columna, np.linspace(0, 100, 20))
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
        self.raiz = self._construir_arbol(X, y.astype(int))
        if np.sum(self.importancias) > 0:
            self.importancias = self.importancias / np.sum(self.importancias)
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
    
    def obtener_arbol_texto(self, nodo=None, profundidad=0, prefijo="", max_prof=4):
        if nodo is None:
            nodo = self.raiz
        texto = ""
        if profundidad > max_prof:
            return prefijo + "[...]\n"
        if nodo.es_hoja():
            clase = "COMPRAR" if nodo.valor == 1 else "NO COMPRAR"
            return prefijo + f"[{clase}]\n"
        nombre = self.nombres_caracteristicas[nodo.caracteristica]
        texto += prefijo + f"{nombre} <= {nodo.umbral:.4f}?\n"
        texto += self.obtener_arbol_texto(nodo.izquierda, profundidad + 1, prefijo + "  SI: ", max_prof)
        texto += self.obtener_arbol_texto(nodo.derecha, profundidad + 1, prefijo + "  NO: ", max_prof)
        return texto


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


# =====================================================================
# SISTEMA PRINCIPAL
# =====================================================================

class TreeSharesInvestment:
    def __init__(self):
        self.arbol_decision = ArbolDecisionManual(max_profundidad=8, min_muestras_division=20)
        self.bst_precios = ArbolBST()
        self.grafo = GrafoCorrelaciones()
        self.datos_acciones = None
        self.modelo_entrenado = False
        self.precios_hist = {}
        self.caracs = ['volatilidad', 'rsi', 'roe', 'pe_ratio', 'margen_ebitda', 'deuda_ebitda', 'volumen']
    
    def cargar_csv(self, ruta_csv='stock_details_5_years.csv'):
        df = pd.read_csv(ruta_csv)
        return df
    
    def preprocesar(self, df, callback=None):
        empresas = df.groupby('Company')
        datos_proc = []
        self.precios_hist = {}
        total = len(empresas)
        
        for i, (ticker, grupo) in enumerate(empresas):
            if callback and i % 50 == 0:
                callback(f"Procesando {i}/{total} empresas...")
            
            grupo = grupo.sort_values('Date')
            precios = grupo['Close'].values
            volumenes = grupo['Volume'].values
            if len(precios) < 50:
                continue
            self.precios_hist[ticker] = precios
            rendimiento = (precios[-1] - precios[0]) / precios[0] if precios[0] != 0 else 0
            volatilidad = np.std(np.diff(precios) / precios[:-1]) if len(precios) > 1 else 0
            np.random.seed(hash(ticker) % 2**32)
            datos_proc.append({
                'ticker': ticker, 'precio': precios[-1], 'rendimiento': rendimiento,
                'volatilidad': volatilidad, 'volumen': np.mean(volumenes),
                'rsi': np.random.uniform(20, 80), 'roe': np.random.uniform(-0.1, 0.4),
                'pe_ratio': np.random.uniform(5, 50), 'margen_ebitda': np.random.uniform(0.05, 0.4),
                'deuda_ebitda': np.random.uniform(0.5, 5)
            })
        
        self.datos_acciones = pd.DataFrame(datos_proc)
        return self.datos_acciones
    
    def construir_bst(self, callback=None):
        for i, row in self.datos_acciones.iterrows():
            self.bst_precios.insertar(row['ticker'], row['precio'], row.to_dict())
            if callback and i % 100 == 0:
                callback(f"BST: {i} acciones insertadas...")
    
    def construir_grafo(self, umbral=0.5, callback=None):
        tickers = list(self.precios_hist.keys())
        for t in tickers:
            datos_t = self.datos_acciones[self.datos_acciones['ticker'] == t]
            if len(datos_t) > 0:
                self.grafo.agregar_vertice(t, datos_t.iloc[0].to_dict())
        
        min_len = min(len(self.precios_hist[t]) for t in tickers) if tickers else 0
        if min_len < 10:
            return
        
        matriz = np.array([self.precios_hist[t][-min_len:] for t in tickers])
        rends = np.diff(matriz, axis=1) / (matriz[:, :-1] + 1e-10)
        
        n_comp = min(len(tickers), 100)
        indices = np.random.choice(len(tickers), n_comp, replace=False)
        
        total_pairs = len(indices) * (len(indices) - 1) // 2
        count = 0
        
        for i, idx_i in enumerate(indices):
            for idx_j in indices[i+1:]:
                count += 1
                if callback and count % 500 == 0:
                    callback(f"Grafo: {count}/{total_pairs} correlaciones...")
                corr = np.corrcoef(rends[idx_i], rends[idx_j])[0, 1]
                if not np.isnan(corr) and abs(corr) >= umbral:
                    self.grafo.agregar_arista(tickers[idx_i], tickers[idx_j], corr)
    
    def calcular_target(self):
        rend_bench = self.datos_acciones['rendimiento'].median()
        self.datos_acciones['supera'] = (self.datos_acciones['rendimiento'] > rend_bench).astype(int)
        return rend_bench
    
    def entrenar(self, callback=None):
        if callback:
            callback("Preparando datos de entrenamiento...")
        
        X = self.datos_acciones[self.caracs].fillna(0)
        y = self.datos_acciones['supera']
        
        n_train = int(len(X) * 0.8)
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        X_train, y_train = X.iloc[indices[:n_train]], y.iloc[indices[:n_train]]
        X_test, y_test = X.iloc[indices[n_train:]], y.iloc[indices[n_train:]]
        
        if callback:
            callback("Construyendo arbol de decision...")
        
        self.arbol_decision.entrenar(X_train, y_train)
        
        preds = self.arbol_decision.predecir(X_test)
        precision = np.mean(preds == y_test.values)
        
        self.modelo_entrenado = True
        
        # Calcular metricas
        y_real = y_test.values
        tp = np.sum((preds == 1) & (y_real == 1))
        tn = np.sum((preds == 0) & (y_real == 0))
        fp = np.sum((preds == 1) & (y_real == 0))
        fn = np.sum((preds == 0) & (y_real == 1))
        
        accuracy = (tp + tn) / len(y_real) if len(y_real) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
        
        return {
            'accuracy': accuracy, 'precision': prec, 'recall': recall, 'f1': f1,
            'profundidad': self.arbol_decision._profundidad_arbol(self.arbol_decision.raiz),
            'nodos': self.arbol_decision._contar_nodos(self.arbol_decision.raiz)
        }
    
    def recomendaciones(self, n=15):
        if not self.modelo_entrenado:
            return []
        X = self.datos_acciones[self.caracs].fillna(0)
        preds = self.arbol_decision.predecir(X)
        self.datos_acciones['pred'] = preds
        recomendadas = self.datos_acciones[self.datos_acciones['pred'] == 1].copy()
        recomendadas = recomendadas.nlargest(n, 'rendimiento')
        return recomendadas.to_dict('records')
    
    def simular_portafolio(self, tickers, capital_inicial=10000):
        portafolio = []
        capital_por_accion = capital_inicial / len(tickers) if tickers else 0
        
        for ticker in tickers:
            datos = self.datos_acciones[self.datos_acciones['ticker'] == ticker]
            if len(datos) == 0:
                continue
            datos = datos.iloc[0]
            precio = datos['precio']
            acciones = capital_por_accion / precio if precio > 0 else 0
            rendimiento = datos['rendimiento']
            valor_final = capital_por_accion * (1 + rendimiento)
            portafolio.append({
                'ticker': ticker, 'precio': precio, 'acciones': acciones,
                'inversion': capital_por_accion, 'rendimiento': rendimiento,
                'valor_final': valor_final, 'ganancia': valor_final - capital_por_accion
            })
        
        if not portafolio:
            return None
        
        total_invertido = sum(p['inversion'] for p in portafolio)
        total_final = sum(p['valor_final'] for p in portafolio)
        rendimiento_total = (total_final - total_invertido) / total_invertido if total_invertido > 0 else 0
        
        return {
            'acciones': portafolio,
            'capital_inicial': capital_inicial,
            'valor_final': total_final,
            'rendimiento_total': rendimiento_total,
            'ganancia_total': total_final - capital_inicial
        }


# =====================================================================
# INTERFAZ GRAFICA
# =====================================================================

class InterfazTreeShares:
    def __init__(self, root):
        self.root = root
        self.root.title("TreeShares Investment - Universidad del Rosario")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        self.sistema = TreeSharesInvestment()
        self.datos_cargados = False
        self.modelo_entrenado = False
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titulo
        titulo = tk.Label(main_frame, text="TREESHARES INVESTMENT", 
                         font=('Arial', 24, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        titulo.pack(pady=10)
        
        subtitulo = tk.Label(main_frame, text="Sistema de Analisis de Inversiones - Universidad del Rosario", 
                            font=('Arial', 12), fg='#888', bg='#1a1a2e')
        subtitulo.pack()
        
        # Frame de estado
        self.estado_frame = tk.Frame(main_frame, bg='#16213e', relief=tk.RIDGE, bd=2)
        self.estado_frame.pack(fill=tk.X, pady=10)
        
        self.lbl_estado = tk.Label(self.estado_frame, text="Estado: Sin datos cargados", 
                                   font=('Arial', 11), fg='#ffd700', bg='#16213e')
        self.lbl_estado.pack(pady=5)
        
        # Notebook para pestanas
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#1a1a2e')
        style.configure('TNotebook.Tab', background='#16213e', foreground='white', padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', '#0f3460')])
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Crear pestanas
        self.crear_tab_datos()
        self.crear_tab_bst()
        self.crear_tab_grafo()
        self.crear_tab_arbol()
        self.crear_tab_portafolio()
        self.crear_tab_recomendaciones()
    
    def crear_tab_datos(self):
        tab = tk.Frame(self.notebook, bg='#16213e')
        self.notebook.add(tab, text='  Datos  ')
        
        # Boton cargar
        btn_frame = tk.Frame(tab, bg='#16213e')
        btn_frame.pack(pady=20)
        
        btn_cargar = tk.Button(btn_frame, text="CARGAR DATOS (CSV)", font=('Arial', 14, 'bold'),
                              bg='#00d4ff', fg='black', width=25, height=2,
                              command=self.cargar_datos)
        btn_cargar.pack(pady=10)
        
        btn_entrenar = tk.Button(btn_frame, text="ENTRENAR MODELO", font=('Arial', 14, 'bold'),
                                bg='#ffd700', fg='black', width=25, height=2,
                                command=self.entrenar_modelo)
        btn_entrenar.pack(pady=10)
        
        # Area de texto para log
        self.txt_log = scrolledtext.ScrolledText(tab, width=100, height=20, 
                                                  bg='#0f0f1a', fg='#00ff00',
                                                  font=('Consolas', 10))
        self.txt_log.pack(pady=10, padx=20)
        
        # Metricas
        self.lbl_metricas = tk.Label(tab, text="", font=('Arial', 11), 
                                     fg='white', bg='#16213e', justify=tk.LEFT)
        self.lbl_metricas.pack(pady=10)
    
    def crear_tab_bst(self):
        tab = tk.Frame(self.notebook, bg='#16213e')
        self.notebook.add(tab, text='  BST (Precios)  ')
        
        # Frame de busqueda
        search_frame = tk.Frame(tab, bg='#16213e')
        search_frame.pack(pady=20)
        
        tk.Label(search_frame, text="Precio Minimo:", font=('Arial', 12), 
                fg='white', bg='#16213e').grid(row=0, column=0, padx=10)
        self.entry_precio_min = tk.Entry(search_frame, font=('Arial', 12), width=15)
        self.entry_precio_min.grid(row=0, column=1, padx=10)
        self.entry_precio_min.insert(0, "50")
        
        tk.Label(search_frame, text="Precio Maximo:", font=('Arial', 12), 
                fg='white', bg='#16213e').grid(row=0, column=2, padx=10)
        self.entry_precio_max = tk.Entry(search_frame, font=('Arial', 12), width=15)
        self.entry_precio_max.grid(row=0, column=3, padx=10)
        self.entry_precio_max.insert(0, "200")
        
        btn_buscar = tk.Button(search_frame, text="BUSCAR", font=('Arial', 12, 'bold'),
                              bg='#00d4ff', fg='black', command=self.buscar_bst)
        btn_buscar.grid(row=0, column=4, padx=20)
        
        # Resultados
        self.txt_bst = scrolledtext.ScrolledText(tab, width=100, height=25, 
                                                  bg='#0f0f1a', fg='#00ff00',
                                                  font=('Consolas', 10))
        self.txt_bst.pack(pady=10, padx=20)
    
    def crear_tab_grafo(self):
        tab = tk.Frame(self.notebook, bg='#16213e')
        self.notebook.add(tab, text='  Grafo (Correlaciones)  ')
        
        # Frame de busqueda
        search_frame = tk.Frame(tab, bg='#16213e')
        search_frame.pack(pady=20)
        
        tk.Label(search_frame, text="Ticker:", font=('Arial', 12), 
                fg='white', bg='#16213e').grid(row=0, column=0, padx=10)
        self.entry_ticker = tk.Entry(search_frame, font=('Arial', 12), width=15)
        self.entry_ticker.grid(row=0, column=1, padx=10)
        self.entry_ticker.insert(0, "AAPL")
        
        btn_analizar = tk.Button(search_frame, text="ANALIZAR CORRELACIONES", 
                                font=('Arial', 12, 'bold'),
                                bg='#00d4ff', fg='black', command=self.analizar_grafo)
        btn_analizar.grid(row=0, column=2, padx=20)
        
        # Resultados
        self.txt_grafo = scrolledtext.ScrolledText(tab, width=100, height=25, 
                                                    bg='#0f0f1a', fg='#00ff00',
                                                    font=('Consolas', 10))
        self.txt_grafo.pack(pady=10, padx=20)
    
    def crear_tab_arbol(self):
        tab = tk.Frame(self.notebook, bg='#16213e')
        self.notebook.add(tab, text='  Arbol Decision  ')
        
        btn_ver = tk.Button(tab, text="VER ARBOL DE DECISION", font=('Arial', 14, 'bold'),
                           bg='#ffd700', fg='black', command=self.ver_arbol)
        btn_ver.pack(pady=20)
        
        self.txt_arbol = scrolledtext.ScrolledText(tab, width=100, height=28, 
                                                    bg='#0f0f1a', fg='#00ff00',
                                                    font=('Consolas', 10))
        self.txt_arbol.pack(pady=10, padx=20)
    
    def crear_tab_portafolio(self):
        tab = tk.Frame(self.notebook, bg='#16213e')
        self.notebook.add(tab, text='  Simular Portafolio  ')
        
        # Frame de entrada
        input_frame = tk.Frame(tab, bg='#16213e')
        input_frame.pack(pady=20)
        
        tk.Label(input_frame, text="Tickers (separados por coma):", font=('Arial', 12), 
                fg='white', bg='#16213e').grid(row=0, column=0, padx=10)
        self.entry_tickers = tk.Entry(input_frame, font=('Arial', 12), width=40)
        self.entry_tickers.grid(row=0, column=1, padx=10)
        self.entry_tickers.insert(0, "AAPL,MSFT,GOOGL,AMZN")
        
        tk.Label(input_frame, text="Capital ($):", font=('Arial', 12), 
                fg='white', bg='#16213e').grid(row=1, column=0, padx=10, pady=10)
        self.entry_capital = tk.Entry(input_frame, font=('Arial', 12), width=20)
        self.entry_capital.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        self.entry_capital.insert(0, "10000")
        
        btn_simular = tk.Button(input_frame, text="SIMULAR PORTAFOLIO", 
                               font=('Arial', 12, 'bold'),
                               bg='#00d4ff', fg='black', command=self.simular_portafolio)
        btn_simular.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Resultados
        self.txt_portafolio = scrolledtext.ScrolledText(tab, width=100, height=22, 
                                                         bg='#0f0f1a', fg='#00ff00',
                                                         font=('Consolas', 10))
        self.txt_portafolio.pack(pady=10, padx=20)
    
    def crear_tab_recomendaciones(self):
        tab = tk.Frame(self.notebook, bg='#16213e')
        self.notebook.add(tab, text='  Recomendaciones  ')
        
        btn_rec = tk.Button(tab, text="OBTENER TOP 15 RECOMENDACIONES", 
                           font=('Arial', 14, 'bold'),
                           bg='#00ff00', fg='black', command=self.ver_recomendaciones)
        btn_rec.pack(pady=20)
        
        self.txt_recomendaciones = scrolledtext.ScrolledText(tab, width=100, height=28, 
                                                              bg='#0f0f1a', fg='#00ff00',
                                                              font=('Consolas', 10))
        self.txt_recomendaciones.pack(pady=10, padx=20)
    
    def log(self, mensaje):
        self.txt_log.insert(tk.END, mensaje + "\n")
        self.txt_log.see(tk.END)
        self.root.update()
    
    def cargar_datos(self):
        self.txt_log.delete(1.0, tk.END)
        self.log("="*60)
        self.log("  CARGANDO DATOS...")
        self.log("="*60)
        
        try:
            self.log("\n[1/5] Leyendo archivo CSV...")
            df = self.sistema.cargar_csv('stock_details_5_years.csv')
            self.log(f"      Archivo cargado: {len(df):,} filas")
            self.log(f"      Empresas unicas: {df['Company'].nunique()}")
            
            self.log("\n[2/5] Preprocesando datos...")
            self.sistema.preprocesar(df, callback=self.log)
            self.log(f"      Procesadas: {len(self.sistema.datos_acciones)} empresas")
            
            self.log("\n[3/5] Calculando target (benchmark)...")
            bench = self.sistema.calcular_target()
            self.log(f"      Benchmark (mediana rendimiento): {bench:.2%}")
            
            self.log("\n[4/5] Construyendo BST...")
            self.sistema.construir_bst(callback=self.log)
            self.log(f"      Nodos BST: {self.sistema.bst_precios.tamano}")
            
            self.log("\n[5/5] Construyendo Grafo de correlaciones...")
            self.sistema.construir_grafo(callback=self.log)
            self.log(f"      Vertices: {self.sistema.grafo.n_vertices}")
            self.log(f"      Aristas: {self.sistema.grafo.n_aristas}")
            
            self.datos_cargados = True
            self.lbl_estado.config(text="Estado: Datos cargados - Listo para entrenar")
            self.log("\n" + "="*60)
            self.log("  DATOS CARGADOS EXITOSAMENTE")
            self.log("="*60)
            
            messagebox.showinfo("Exito", "Datos cargados correctamente!")
            
        except FileNotFoundError:
            self.log("\nERROR: No se encontro el archivo stock_details_5_years.csv")
            messagebox.showerror("Error", "No se encontro el archivo CSV")
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def entrenar_modelo(self):
        if not self.datos_cargados:
            messagebox.showwarning("Aviso", "Primero debes cargar los datos")
            return
        
        self.log("\n" + "="*60)
        self.log("  ENTRENANDO MODELO...")
        self.log("="*60)
        
        try:
            metricas = self.sistema.entrenar(callback=self.log)
            
            self.log(f"\n  Arbol construido!")
            self.log(f"  Profundidad: {metricas['profundidad']}")
            self.log(f"  Nodos: {metricas['nodos']}")
            
            self.modelo_entrenado = True
            self.lbl_estado.config(text="Estado: Modelo entrenado - Sistema listo")
            
            texto_metricas = f"""
METRICAS DEL MODELO:
  Accuracy:  {metricas['accuracy']*100:.2f}%
  Precision: {metricas['precision']*100:.2f}%
  Recall:    {metricas['recall']*100:.2f}%
  F1-Score:  {metricas['f1']*100:.2f}%
"""
            self.lbl_metricas.config(text=texto_metricas)
            self.log(texto_metricas)
            
            messagebox.showinfo("Exito", f"Modelo entrenado!\nAccuracy: {metricas['accuracy']*100:.1f}%")
            
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def buscar_bst(self):
        if not self.datos_cargados:
            messagebox.showwarning("Aviso", "Primero debes cargar los datos")
            return
        
        try:
            precio_min = float(self.entry_precio_min.get())
            precio_max = float(self.entry_precio_max.get())
        except:
            messagebox.showerror("Error", "Ingresa valores numericos validos")
            return
        
        self.txt_bst.delete(1.0, tk.END)
        resultados = self.sistema.bst_precios.buscar_por_rango(precio_min, precio_max)
        
        self.txt_bst.insert(tk.END, "="*70 + "\n")
        self.txt_bst.insert(tk.END, f"  BUSQUEDA BST: ${precio_min:.2f} - ${precio_max:.2f}\n")
        self.txt_bst.insert(tk.END, f"  Encontradas: {len(resultados)} acciones\n")
        self.txt_bst.insert(tk.END, "="*70 + "\n\n")
        
        self.txt_bst.insert(tk.END, f"{'TICKER':<10} {'PRECIO':>12} {'RENDIMIENTO':>15}\n")
        self.txt_bst.insert(tk.END, "-"*40 + "\n")
        
        for r in sorted(resultados, key=lambda x: x['precio'], reverse=True)[:50]:
            rend = r['datos'].get('rendimiento', 0) if r['datos'] else 0
            self.txt_bst.insert(tk.END, f"{r['ticker']:<10} ${r['precio']:>10.2f} {rend*100:>14.1f}%\n")
    
    def analizar_grafo(self):
        if not self.datos_cargados:
            messagebox.showwarning("Aviso", "Primero debes cargar los datos")
            return
        
        ticker = self.entry_ticker.get().strip().upper()
        
        if ticker not in self.sistema.grafo.vertices:
            messagebox.showerror("Error", f"Ticker '{ticker}' no encontrado")
            return
        
        self.txt_grafo.delete(1.0, tk.END)
        
        self.txt_grafo.insert(tk.END, "="*70 + "\n")
        self.txt_grafo.insert(tk.END, f"  ANALISIS DE CORRELACIONES: {ticker}\n")
        self.txt_grafo.insert(tk.END, "="*70 + "\n\n")
        
        # Similares
        similares = self.sistema.grafo.encontrar_similares(ticker)[:10]
        self.txt_grafo.insert(tk.END, "ACCIONES SIMILARES (alta correlacion):\n")
        self.txt_grafo.insert(tk.END, "-"*40 + "\n")
        if similares:
            for v, c in similares:
                self.txt_grafo.insert(tk.END, f"  {v}: {c:.3f}\n")
        else:
            self.txt_grafo.insert(tk.END, "  No se encontraron acciones similares\n")
        
        # Diversificadas
        self.txt_grafo.insert(tk.END, "\nACCIONES PARA DIVERSIFICAR (baja correlacion):\n")
        self.txt_grafo.insert(tk.END, "-"*40 + "\n")
        diversas = self.sistema.grafo.encontrar_diversificadas(ticker)[:10]
        if diversas:
            for v, c in diversas:
                self.txt_grafo.insert(tk.END, f"  {v}: {c:.3f}\n")
        else:
            self.txt_grafo.insert(tk.END, "  No se encontraron acciones para diversificar\n")
        
        # BFS
        self.txt_grafo.insert(tk.END, "\nACCIONES RELACIONADAS (BFS profundidad 2):\n")
        self.txt_grafo.insert(tk.END, "-"*40 + "\n")
        relacionadas = self.sistema.grafo.bfs(ticker, max_prof=2)[:15]
        if relacionadas:
            for r in relacionadas:
                self.txt_grafo.insert(tk.END, f"  {r['ticker']} (profundidad {r['profundidad']})\n")
        else:
            self.txt_grafo.insert(tk.END, "  No se encontraron acciones relacionadas\n")
    
    def ver_arbol(self):
        if not self.modelo_entrenado:
            messagebox.showwarning("Aviso", "Primero debes entrenar el modelo")
            return
        
        self.txt_arbol.delete(1.0, tk.END)
        
        self.txt_arbol.insert(tk.END, "="*70 + "\n")
        self.txt_arbol.insert(tk.END, "  ARBOL DE DECISION\n")
        self.txt_arbol.insert(tk.END, "="*70 + "\n\n")
        
        prof = self.sistema.arbol_decision._profundidad_arbol(self.sistema.arbol_decision.raiz)
        nodos = self.sistema.arbol_decision._contar_nodos(self.sistema.arbol_decision.raiz)
        
        self.txt_arbol.insert(tk.END, f"Profundidad: {prof}\n")
        self.txt_arbol.insert(tk.END, f"Nodos totales: {nodos}\n\n")
        self.txt_arbol.insert(tk.END, "ESTRUCTURA DEL ARBOL:\n")
        self.txt_arbol.insert(tk.END, "-"*50 + "\n\n")
        
        texto_arbol = self.sistema.arbol_decision.obtener_arbol_texto()
        self.txt_arbol.insert(tk.END, texto_arbol)
    
    def simular_portafolio(self):
        if not self.datos_cargados:
            messagebox.showwarning("Aviso", "Primero debes cargar los datos")
            return
        
        tickers_str = self.entry_tickers.get().strip().upper()
        tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
        
        try:
            capital = float(self.entry_capital.get())
        except:
            capital = 10000
        
        resultado = self.sistema.simular_portafolio(tickers, capital)
        
        self.txt_portafolio.delete(1.0, tk.END)
        
        if not resultado or not resultado['acciones']:
            self.txt_portafolio.insert(tk.END, "No se encontraron las acciones especificadas")
            return
        
        self.txt_portafolio.insert(tk.END, "="*70 + "\n")
        self.txt_portafolio.insert(tk.END, "  SIMULACION DE PORTAFOLIO\n")
        self.txt_portafolio.insert(tk.END, "="*70 + "\n\n")
        
        self.txt_portafolio.insert(tk.END, f"{'TICKER':<8} {'PRECIO':>10} {'ACCIONES':>10} {'INVERSION':>12} {'REND':>8} {'VALOR FINAL':>12}\n")
        self.txt_portafolio.insert(tk.END, "-"*65 + "\n")
        
        for acc in resultado['acciones']:
            self.txt_portafolio.insert(tk.END, 
                f"{acc['ticker']:<8} ${acc['precio']:>8.2f} {acc['acciones']:>10.2f} ${acc['inversion']:>10.2f} {acc['rendimiento']*100:>7.1f}% ${acc['valor_final']:>10.2f}\n")
        
        self.txt_portafolio.insert(tk.END, "-"*65 + "\n\n")
        self.txt_portafolio.insert(tk.END, "RESUMEN:\n")
        self.txt_portafolio.insert(tk.END, f"  Capital Inicial:    ${resultado['capital_inicial']:>12,.2f}\n")
        self.txt_portafolio.insert(tk.END, f"  Valor Final:        ${resultado['valor_final']:>12,.2f}\n")
        self.txt_portafolio.insert(tk.END, f"  Ganancia/Perdida:   ${resultado['ganancia_total']:>12,.2f}\n")
        self.txt_portafolio.insert(tk.END, f"  Rendimiento Total:  {resultado['rendimiento_total']*100:>12.2f}%\n")
    
    def ver_recomendaciones(self):
        if not self.modelo_entrenado:
            messagebox.showwarning("Aviso", "Primero debes entrenar el modelo")
            return
        
        recomendaciones = self.sistema.recomendaciones(15)
        
        self.txt_recomendaciones.delete(1.0, tk.END)
        
        self.txt_recomendaciones.insert(tk.END, "="*70 + "\n")
        self.txt_recomendaciones.insert(tk.END, "  TOP 15 RECOMENDACIONES DE COMPRA\n")
        self.txt_recomendaciones.insert(tk.END, "="*70 + "\n\n")
        
        self.txt_recomendaciones.insert(tk.END, f"{'#':<4} {'TICKER':<10} {'PRECIO':>12} {'RENDIMIENTO':>15}\n")
        self.txt_recomendaciones.insert(tk.END, "-"*45 + "\n")
        
        for i, r in enumerate(recomendaciones, 1):
            self.txt_recomendaciones.insert(tk.END, 
                f"{i:<4} {r['ticker']:<10} ${r['precio']:>10.2f} {r['rendimiento']*100:>14.1f}%\n")
        
        self.txt_recomendaciones.insert(tk.END, "\n" + "="*70 + "\n")
        self.txt_recomendaciones.insert(tk.END, "  Estas acciones fueron predichas como 'SUPERA BENCHMARK'\n")
        self.txt_recomendaciones.insert(tk.END, "  por el Arbol de Decision entrenado.\n")
        self.txt_recomendaciones.insert(tk.END, "="*70 + "\n")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazTreeShares(root)
    root.mainloop()
