# TREE SHARES: PROYECTO FINAL ESTRUCTURAS DE DATOS
# Implementación manual de: Grafo ponderado, BST, Árbol de Decisión para análisis de acciones financieras

import pandas as pd

# ==== PARTE 1: ESTRUCTURAS DE DATOS ====

# --- Grafo No Dirigido Ponderado ---
class Grafo:
    def __init__(self):
        self.adyacentes = {}  # ticker: lista de (vecino, peso)

    def agregar_nodo(self, ticker):
        if ticker not in self.adyacentes:
            self.adyacentes[ticker] = []

    def agregar_arista(self, ticker1, ticker2, peso):
        self.agregar_nodo(ticker1)
        self.agregar_nodo(ticker2)
        self.adyacentes[ticker1].append((ticker2, peso))
        self.adyacentes[ticker2].append((ticker1, peso))  # No dirigido

    def bfs(self, inicio, max_profundidad=1):
        visitados = {inicio}
        cola = [(inicio, 0)]
        resultado = []
        while cola:
            actual, profundidad = cola.pop(0)
            resultado.append(actual)
            if profundidad < max_profundidad:
                for vecino, _ in self.adyacentes.get(actual, []):
                    if vecino not in visitados:
                        visitados.add(vecino)
                        cola.append((vecino, profundidad+1))
        return resultado

    def diversificar_portafolio(self, candidatos, n):
        # Algoritmo greedy: Elegir la acción menos conectada con las ya elegidas
        if not candidatos: return []
        portafolio = [candidatos[0]]
        while len(portafolio) < n and len(portafolio) < len(candidatos):
            min_corr = float('inf')
            mejor = None
            for cand in candidatos:
                if cand in portafolio: continue
                corr = 0
                for elegida in portafolio:
                    peso = next((p for (v, p) in self.adyacentes[cand] if v == elegida), 0)
                    corr += peso
                avg_corr = corr / len(portafolio) if portafolio else 0
                if avg_corr < min_corr:
                    min_corr = avg_corr
                    mejor = cand
            if mejor: portafolio.append(mejor)
            else: break
        return portafolio

# --- Árbol Binario de Búsqueda ---
class NodoBST:
    def __init__(self, ticker, precio, extra={}):
        self.ticker = ticker
        self.precio = precio
        self.extra = extra  # Diccionario con otros datos
        self.izq = None
        self.der = None

class BST:
    def __init__(self):
        self.raiz = None

    def insertar(self, ticker, precio, extra={}):
        def _ins(nodo, ticker, precio, extra):
            if nodo is None:
                return NodoBST(ticker, precio, extra)
            if precio < nodo.precio:
                nodo.izq = _ins(nodo.izq, ticker, precio, extra)
            else:
                nodo.der = _ins(nodo.der, ticker, precio, extra)
            return nodo
        self.raiz = _ins(self.raiz, ticker, precio, extra)

    def rango(self, p_min, p_max):
        resultado = []
        def _rango(nodo):
            if nodo is None:
                return
            if p_min <= nodo.precio <= p_max:
                resultado.append(nodo)
            if nodo.precio > p_min:
                _rango(nodo.izq)
            if nodo.precio < p_max:
                _rango(nodo.der)
        _rango(self.raiz)
        return resultado

# --- Árbol de Decisión Binario ---
class NodoDecision:
    def __init__(self, feature=None, umbral=None, izquierda=None, derecha=None, pred=None):
        self.feature = feature      # idx del feature a usar
        self.umbral = umbral       # el umbral para el split
        self.izquierda = izquierda # subárbol
        self.derecha = derecha
        self.pred = pred           # Solo en hojas: 0 o 1

class DecisionTree:
    def __init__(self, max_depth=4, min_muestras=4):
        self.max_depth = max_depth
        self.min_muestras = min_muestras
        self.raiz = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_muestras or depth == self.max_depth:
            major = max([(y.count(c), c) for c in set(y)])[1]
            return NodoDecision(pred=major)

        n_features = len(X[0])
        mejor_idx, mejor_umbral, mayor_gain = None, None, 0
        base_gini = self._gini(y)
        for f in range(n_features):
            valores = sorted(set(row[f] for row in X))
            for v in valores:
                izq = [yi for xi, yi in zip(X, y) if xi[f] < v]
                der = [yi for xi, yi in zip(X, y) if xi[f] >= v]
                if len(izq)==0 or len(der)==0: continue
                gini_izq = self._gini(izq)
                gini_der = self._gini(der)
                peso_izq = len(izq)/len(y)
                peso_der = len(der)/len(y)
                gini_split = peso_izq*gini_izq + peso_der*gini_der
                gain = base_gini - gini_split
                if gain > mayor_gain:
                    mejor_idx = f
                    mejor_umbral = v
                    mayor_gain = gain
        if mayor_gain == 0 or mejor_idx is None:
            major = max([(y.count(c), c) for c in set(y)])[1]
            return NodoDecision(pred=major)

        X_izq = [xi for xi in X if xi[mejor_idx] < mejor_umbral]
        y_izq = [yi for xi, yi in zip(X, y) if xi[mejor_idx] < mejor_umbral]
        X_der = [xi for xi in X if xi[mejor_idx] >= mejor_umbral]
        y_der = [yi for xi, yi in zip(X, y) if xi[mejor_idx] >= mejor_umbral]
        izquierda = self.fit(X_izq, y_izq, depth+1)
        derecha = self.fit(X_der, y_der, depth+1)
        nodo = NodoDecision(feature=mejor_idx, umbral=mejor_umbral, izquierda=izquierda, derecha=derecha)
        if depth == 0:
            self.raiz = nodo
        return nodo

    def _gini(self, etiquetas):
        total = len(etiquetas)
        if total == 0: return 0
        counts = [etiquetas.count(c)/total for c in set(etiquetas)]
        return 1 - sum(c**2 for c in counts)

    def predict_una(self, x, nodo=None):
        if nodo is None: nodo = self.raiz
        if nodo.pred is not None:
            return nodo.pred
        if x[nodo.feature] < nodo.umbral:
            return self.predict_una(x, nodo.izquierda)
        else:
            return self.predict_una(x, nodo.derecha)

    def predict(self, X):
        return [self.predict_una(xi) for xi in X]

# ==== PARTE 2: LECTURA Y PREPARACIÓN DE DATOS ====
def cargar_acciones_simple(data_path, n=150):
    df = pd.read_csv(data_path)
    campos = ['Date','Close','Volume','Company']
    for c in campos:
        if c not in df.columns:
            raise Exception(f"Falta columna {c} en el CSV")
    acciones = []
    for company, grupo in df.groupby('Company'):
        if len(grupo)<100: continue
        grupo = grupo.sort_values("Date")
        ultima = grupo.iloc[-1]
        precio = float(ultima['Close'])
        volumen = float(ultima['Volume'])
        inicio = grupo.iloc[0]['Close']
        y = 1 if (precio-inicio)>0 else 0
        acciones.append({
            "ticker": company,
            "precio": precio,
            "features": [precio, volumen],
            "y": y
        })
        if len(acciones)>=n:
            break
    return acciones


# ==== PARTE 3: MENÚ Y FLUJO DE USUARIO ====
def main():
    print("====== TREE SHARES INVESTMENT SYSTEM ======")
    print("Cargando datos, esto puede tomar algunos segundos...")
    DATA_PATH = "stock_details_5_years.csv"  # Ajusta al path real de tu archivo
    acciones = cargar_acciones_simple(DATA_PATH, n=100)

    # Crear estructuras
    bst = BST()
    grafo = Grafo()
    X, Y = [], []

    print(f"\nInsertando {len(acciones)} acciones en BST y armando dataset...")
    for a in acciones:
        bst.insertar(a["ticker"], a["precio"], extra=a)
        X.append(a["features"])
        Y.append(a["y"])

    print("Creando correlaciones de ejemplo...")
    for a in acciones:
        for b in acciones:
            if a==b: continue
            diff = abs(a['precio']-b['precio'])
            corr = 1.0 if diff<10 else (0.7 if diff<30 else 0.2)
            grafo.agregar_arista(a["ticker"], b["ticker"], corr)

    arbol = DecisionTree()
    print("Entrenando árbol de decisión para predicción...")
    arbol.fit(X, Y)

    while True:
        print("\n---- MENÚ TREE SHARES ----")
        print("1. Buscar acciones por rango de precio")
        print("2. Predecir si una acción supera el benchmark")
        print("3. Sugerir portafolio diversificado")
        print("4. Salir")
        op = input("Elegir una opción: ").strip()
        
        if op=='1':
            pmin = float(input("Precio mínimo: "))
            pmax = float(input("Precio máximo: "))
            nodos = bst.rango(pmin, pmax)
            print(f"{len(nodos)} acciones encontradas:")
            for nodo in nodos:
                print(f"{nodo.ticker}: ${nodo.precio:.2f}")
        elif op=='2':
            t = input("Ticker de la acción: ").strip().upper()
            nodos = bst.rango(0, float('inf'))
            nodo = next((n for n in nodos if n.ticker==t), None)
            if not nodo:
                print("Ticker no encontrado.")
            else:
                pred = arbol.predict_una(nodo.extra["features"])
                print(f"{nodo.ticker}: {'SUPERA el benchmark' if pred==1 else 'NO supera el benchmark'}")
        elif op=='3':
            pmin = float(input("Precio mínimo: "))
            pmax = float(input("Precio máximo: "))
            n = int(input("¿Cuántas acciones en el portafolio?: "))
            nodos = bst.rango(pmin, pmax)
            aprobadas = []
            for nodo in nodos:
                if arbol.predict_una(nodo.extra["features"]):
                    aprobadas.append(nodo.ticker)
            sugeridas = grafo.diversificar_portafolio(aprobadas, n)
            print(f"Portafolio recomendado: {sugeridas}")
        elif op=='4':
            print("¡Gracias por usar TreeShares!")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()


