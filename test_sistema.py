# -*- coding: utf-8 -*-
"""
TREESHARES INVESTMENT - PRUEBAS DEL SISTEMA
Universidad del Rosario - Algoritmos y Estructuras de Datos

Integrantes:
- Carlos Gutierrez
- Samuel Valderrama  
- David Pascagaza
"""

import numpy as np
import pandas as pd
import os

from interfaz_grafica import (
    ArbolDecisionManual,
    ArbolBST,
    GrafoCorrelaciones,
    TreeSharesInvestment
)


def prueba_pasada(nombre):
    print(f"  [OK] {nombre}")

def prueba_fallida(nombre, error):
    print(f"  [X] {nombre}: {error}")


# ============================================================
# PRUEBAS DEL ARBOL DE DECISION
# ============================================================

def test_arbol_decision():
    print("\n" + "="*50)
    print("  PRUEBAS: ARBOL DE DECISION")
    print("="*50)
    
    pasadas = 0
    
    # Test 1: Crear arbol
    try:
        arbol = ArbolDecisionManual(max_profundidad=5)
        assert arbol.raiz is None
        prueba_pasada("Crear arbol vacio")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Crear arbol", e)
    
    # Test 2: Entropia
    try:
        arbol = ArbolDecisionManual()
        y = np.array([0, 0, 1, 1])
        entropia = arbol._calcular_entropia(y)
        assert 0.9 < entropia < 1.1  # Entropia maxima ~1
        prueba_pasada("Calcular entropia")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Entropia", e)
    
    # Test 3: Entrenar y predecir
    try:
        arbol = ArbolDecisionManual(max_profundidad=5)
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = (X[:, 0] > 0.5).astype(int)
        
        arbol.entrenar(X, y)
        predicciones = arbol.predecir(X)
        
        accuracy = np.mean(predicciones == y)
        assert accuracy > 0.8
        prueba_pasada(f"Entrenar y predecir (acc: {accuracy:.0%})")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Entrenar", e)
    
    # Test 4: Profundidad maxima
    try:
        arbol = ArbolDecisionManual(max_profundidad=3)
        X = np.random.rand(50, 2)
        y = (X[:, 0] > 0.5).astype(int)
        arbol.entrenar(X, y)
        
        prof = arbol._profundidad_arbol(arbol.raiz)
        assert prof <= 3
        prueba_pasada(f"Profundidad controlada ({prof})")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Profundidad", e)
    
    # Test 5: Division train/test 80-20
    try:
        np.random.seed(42)
        n_total = 100
        X = np.random.rand(n_total, 3)
        y = (X[:, 0] > 0.5).astype(int)
        
        n_train = int(n_total * 0.8)
        indices = np.random.permutation(n_total)
        
        X_train = X[indices[:n_train]]
        X_test = X[indices[n_train:]]
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        prueba_pasada("Division train/test 80-20")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Division", e)
    
    # Test 6: Overfitting (train vs test accuracy)
    try:
        arbol = ArbolDecisionManual(max_profundidad=8)
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        # Dividir datos
        X_train, y_train = X[:160], y[:160]
        X_test, y_test = X[160:], y[160:]
        
        arbol.entrenar(X_train, y_train)
        
        acc_train = np.mean(arbol.predecir(X_train) == y_train)
        acc_test = np.mean(arbol.predecir(X_test) == y_test)
        diff = acc_train - acc_test
        
        # Si diferencia > 20% hay overfitting severo
        assert diff < 0.20, f"Overfitting: train={acc_train:.0%}, test={acc_test:.0%}"
        prueba_pasada(f"Sin overfitting (train:{acc_train:.0%}, test:{acc_test:.0%})")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Overfitting", e)
    
    # Test 7: Funciona con diferentes tamaños de datos
    try:
        for n_datos in [50, 500, 2000]:
            arbol = ArbolDecisionManual(max_profundidad=5)
            X = np.random.rand(n_datos, 3)
            y = (X[:, 0] > 0.5).astype(int)
            arbol.entrenar(X, y)
            assert arbol.raiz is not None
        
        prueba_pasada("Funciona con 50, 500, 2000 datos")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Tamaño datos", e)
    
    return pasadas


# ============================================================
# PRUEBAS DEL ARBOL BST
# ============================================================

def test_arbol_bst():
    print("\n" + "="*50)
    print("  PRUEBAS: ARBOL BST")
    print("="*50)
    
    pasadas = 0
    
    # Test 1: Insertar
    try:
        bst = ArbolBST()
        bst.insertar("AAPL", 150)
        bst.insertar("MSFT", 300)
        bst.insertar("GOOGL", 100)
        
        assert bst.tamano == 3
        prueba_pasada("Insertar nodos")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Insertar", e)
    
    # Test 2: Buscar por rango
    try:
        bst = ArbolBST()
        for ticker, precio in [("A", 50), ("B", 100), ("C", 150), ("D", 200)]:
            bst.insertar(ticker, precio)
        
        resultados = bst.buscar_por_rango(100, 150)
        assert len(resultados) == 2  # B y C
        prueba_pasada("Busqueda por rango")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Busqueda", e)
    
    # Test 3: Minimo y maximo
    try:
        bst = ArbolBST()
        bst.insertar("MIN", 10)
        bst.insertar("MAX", 100)
        bst.insertar("MED", 50)
        
        assert bst.obtener_minimo()['precio'] == 10
        assert bst.obtener_maximo()['precio'] == 100
        prueba_pasada("Minimo y maximo")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Min/Max", e)
    
    # Test 4: Insercion masiva
    try:
        bst = ArbolBST()
        for i in range(500):
            bst.insertar(f"S{i}", np.random.uniform(10, 500))
        
        assert bst.tamano == 500
        prueba_pasada("500 inserciones")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Masiva", e)
    
    return pasadas


# ============================================================
# PRUEBAS DEL GRAFO
# ============================================================

def test_grafo():
    print("\n" + "="*50)
    print("  PRUEBAS: GRAFO DE CORRELACIONES")
    print("="*50)
    
    pasadas = 0
    
    # Test 1: Agregar vertices y aristas
    try:
        grafo = GrafoCorrelaciones()
        grafo.agregar_vertice("AAPL")
        grafo.agregar_vertice("MSFT")
        grafo.agregar_arista("AAPL", "MSFT", 0.85)
        
        assert grafo.n_vertices == 2
        assert grafo.n_aristas == 1
        prueba_pasada("Vertices y aristas")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Vertices", e)
    
    # Test 2: Encontrar similares
    try:
        grafo = GrafoCorrelaciones()
        for t in ["A", "B", "C", "D"]:
            grafo.agregar_vertice(t)
        
        grafo.agregar_arista("A", "B", 0.9)
        grafo.agregar_arista("A", "C", 0.8)
        grafo.agregar_arista("A", "D", 0.3)
        
        similares = grafo.encontrar_similares("A", umbral=0.7)
        assert len(similares) == 2  # B y C
        prueba_pasada("Encontrar similares")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Similares", e)
    
    # Test 3: BFS
    try:
        grafo = GrafoCorrelaciones()
        for t in ["A", "B", "C", "D"]:
            grafo.agregar_vertice(t)
        
        grafo.agregar_arista("A", "B", 0.8)
        grafo.agregar_arista("B", "C", 0.8)
        grafo.agregar_arista("C", "D", 0.8)
        
        resultado = grafo.bfs("A", max_prof=2)
        tickers = [r['ticker'] for r in resultado]
        
        assert "B" in tickers and "C" in tickers
        assert "D" not in tickers  # profundidad 3
        prueba_pasada("BFS profundidad 2")
        pasadas += 1
    except Exception as e:
        prueba_fallida("BFS", e)
    
    return pasadas


# ============================================================
# PRUEBAS DE INTEGRACION
# ============================================================

def test_sistema():
    print("\n" + "="*50)
    print("  PRUEBAS: SISTEMA COMPLETO")
    print("="*50)
    
    pasadas = 0
    
    if not os.path.exists('stock_details_5_years.csv'):
        print("  [!] Archivo CSV no encontrado")
        return 0
    
    # Test 1: Cargar datos
    try:
        sistema = TreeSharesInvestment()
        df = sistema.cargar_csv('stock_details_5_years.csv')
        
        assert len(df) > 0
        prueba_pasada(f"Cargar CSV ({len(df):,} filas)")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Cargar CSV", e)
    
    # Test 2: Preprocesar
    try:
        sistema.preprocesar(df)
        assert len(sistema.datos_acciones) > 0
        prueba_pasada(f"Preprocesar ({len(sistema.datos_acciones)} empresas)")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Preprocesar", e)
    
    # Test 3: Construir estructuras
    try:
        sistema.construir_bst()
        sistema.construir_grafo()
        
        assert sistema.bst_precios.tamano > 0
        assert sistema.grafo.n_vertices > 0
        prueba_pasada("Construir BST y Grafo")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Estructuras", e)
    
    # Test 4: Entrenar modelo
    try:
        sistema.calcular_target()
        metricas = sistema.entrenar()
        
        assert sistema.modelo_entrenado == True
        assert metricas['accuracy'] > 0
        prueba_pasada(f"Entrenar (acc: {metricas['accuracy']:.0%})")
        pasadas += 1
    except Exception as e:
        prueba_fallida("Entrenar", e)
    
    return pasadas


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TREESHARES - PRUEBAS DEL SISTEMA")
    print("  Universidad del Rosario")
    print("="*50)
    
    total = 0
    
    total += test_arbol_decision()
    total += test_arbol_bst()
    total += test_grafo()
    total += test_sistema()
    
    print("\n" + "="*50)
    print(f"  TOTAL: {total} pruebas pasadas")
    print("="*50 + "\n")
