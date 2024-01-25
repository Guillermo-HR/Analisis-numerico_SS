# Autor: Guillermo Hernández Ruiz de Esparza
# ------------------- Librerías -------------------
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import pandas as pd
import datetime
import os
from IPython.display import display, Math, Latex

# ------------------- Constantes -------------------
colores = ['blue', 'red', 'green', 'orange', 'purple', 'magenta', 'yellow', 'black']
maxIteraciones = 20

# ------------------- Calculo de errores -------------------
def errorRelativo(valorReal: float, valorAproximado: float) -> float:
    """
    Función para calcular el error relativo porcentual.

    Parámetros:
        valorReal: valor real de la variable.
        valorAproximado: valor aproximado de la variable.

    Retorna:
        error relativo porcentual.
    """
    return abs((valorReal - valorAproximado) / valorReal) * 100

def errorAbsoluto(valorReal: float, valorAproximado: float) -> float:
    """
    Función para calcular el error absoluto.

    Parámetros:
        valorReal: valor real de la variable.
        valorAproximado: valor aproximado de la variable.

    Retorna:
        error absoluto.
    """
    return abs(valorReal - valorAproximado)

# ------------------- Graficación -------------------
def graficar(titulo: str, *args: list, **kwargs: dict)->plt.Figure:
    """
    Función para graficar curvas.

    Parámetros:
        titulo: titulo de la grafica
        
        args: lista de listas con los valores de x, y, label.

        kwargs: diccionario para los valores de facecolor, xlabel, ylabel, ylim, figsize, colores y puntos.
    """
    # Verificar los argumentos
    if len(args) == 0:
        raise ValueError('Se debe ingresar al menos una lista de datos.')
    # Crear una figura
    if 'figsize' in kwargs:
        fig, ax1 = plt.subplots(1, 1, figsize = kwargs['figsize'])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize = (13, 7))
    # Configuración de la gráfica
    ax1.set_title(titulo, fontsize = 25)
    ax1.grid()
    if 'facecolor' in kwargs:
        ax1.set_facecolor(kwargs['facecolor'])
    else:
        ax1.set_facecolor('white')
    if 'xlabel' in kwargs:
        ax1.set_xlabel(kwargs['xlabel'], fontsize=20)
    else:
        ax1.set_xlabel('x', fontsize=20)
    if 'ylabel' in kwargs:
        ax1.set_ylabel(kwargs['ylabel'], fontsize=20)
    else:
        ax1.set_ylabel('y', fontsize=20)
    if 'ylim' in kwargs:
        ax1.set_ylim(kwargs['ylim'])
    else:
        pass
    if 'colores' in kwargs:
        colores_grafica = kwargs['colores']
    else:
        colores_grafica = colores
    if 'markersize' in kwargs:
        tamaño_puntos = kwargs['markersize']
    else:
        tamaño_puntos = 10
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
    else:
        fontsize = 15
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Graficar
    for i, curva in enumerate(args):
        if curva[2] =='':
            if len(curva) == 4:
                ax1.plot(curva[0], curva[1], color = colores_grafica[i], linestyle = curva[3])
            else:
                ax1.plot(curva[0], curva[1], color = colores_grafica[i])
        else:
            if len(curva) == 4:
                ax1.plot(curva[0], curva[1], label = curva[2], color = colores_grafica[i], linestyle = curva[3])
            else:
                ax1.plot(curva[0], curva[1], label = curva[2], color = colores_grafica[i])
    # Marcar puntos de interés
    if 'puntos' in kwargs:
        for punto in kwargs['puntos']:
            ax1.plot(punto[0], punto[1], marker = 'o', markersize = tamaño_puntos, label = punto[2])
    ax1.legend(fontsize = fontsize)
    plt.tight_layout()
    plt.show()
    return fig

def graficarBiseccion(f:sp.Expr, x_i:int, x_s:int)->plt.Figure:
    """
    Funcion para graficar el método de bisección.
    Parametros:
        f: funcion a evaluar.
        x_i: valor inferior del intervalo.
        x_s: valor superior del intervalo.
    Retorna:
        fig: figura de la grafica.
    """
    # crear el rango de x y valores de y
    rango_x = np.linspace(x_i, x_s, 500, dtype = 'float')
    valores_y = [f(x) for x in rango_x]
    fig = graficar('Método de bisección', [rango_x, valores_y, f'f(x)'])
    return fig

def graficarN_R(f:sp.Expr, x_i:float)->plt.Figure:
    """
    Funcion para graficar el método Newton-Raphson.
    Parametros:
        f: funcion a evaluar.
        xi: aproximación de la raíz.
    Retorna:
        fig: figura de la grafica.
    """
    # crear el rango de x y valores de y
    x_i = float(x_i)
    minimo = x_i - 5
    maximo = x_i + 5
    rango_x = np.linspace(minimo, maximo, 500, dtype = 'float')
    valores_y = [f(x) for x in rango_x]

    fig = graficar('Método de Newton-Raphson', [rango_x, valores_y, f'f(x)'], puntos = [(x_i, 0, f'Raíz = {np.round(x_i, 3)}')], markersize = 5, fontsize = 10)
    return fig

def graficarTrapecio(f:sp.Expr, a:float, b:float)->plt.Figure:
    """
    Funcion para graficar el método del trapecio.
    Parametros:
        f: funcion a evaluar.
        a: valor inferior del intervalo.
        b: valor superior del intervalo.
    Retorna:
        fig: figura de la grafica.
    """
    # crear el rango de x y valores de y
    a = float(a)
    b = float(b)
    minimo = a - 1
    maximo = b + 1
    rango_x = np.linspace(minimo, maximo, 500, dtype = 'float')
    valores_y = [f(x) for x in rango_x]

    fig = graficar('Método del trapecio', [rango_x, valores_y, f'f(x)'], puntos = [(a, 0, f'a = {np.round(a, 3)}'), (b, 0, f'b = {np.round(b, 3)}')], markersize = 5, fontsize = 10)
    return fig
# ------------------- Tabulación -------------------

# ------------------- Guardar resultados -------------------

# ------------------- Mostrar resultados -------------------
def mostrarPolinomio(a:list):
    """
    Funcion para mostrar un polinomio.
    Parametros:
        a: lista de coeficientes del polinomio. -> [a_0, a_1, ..., a_n]
    Retorna:
        polinomio: sympy.Poly del polinomio. -> [a_0*x**n + a_1*x**(n-1) + ... + a_n]
    """
    polinomio = ''
    grad = len(a) - 1
    for i, coeficiente in enumerate(a[:-2]):
        if coeficiente != 0:
            polinomio += f'{np.round(coeficiente, 3)}*x**{grad - i} + '
    polinomio += f'{np.round(a[-2], 3)}*x + '
    polinomio += f'{np.round(a[-1], 3)}'
    polinomio = polinomio.replace('+ -', '- ')
    x = sp.Symbol('x')
    polinomio = sp.Poly(polinomio, x)
    display(Math(f'P_{grad}(x) = ' + sp.latex(polinomio.as_expr())))

def mostrarEncabezado(a:list):
    print(f'Iteración\tp\t\tq\t\t', end='')
    for k in range(len(a) - 2):
        print(f'b_{k}', end='\t\t')
    print(f'R\t\tS\t\t\u0394p\t\t\u0394q')
        
def mostrarIteracion(i:int, p:int, q:int, b:list, R:int, S:int, dp:int, dq:int):
    redondeo = 4
    print(f'{np.round(i, redondeo)}\t\t{np.round(p, redondeo)}\t\t{np.round(q, redondeo)}\t\t', end='')
    for k in range(len(b)):
        print(f'{np.round(b[k], redondeo)}\t\t', end='')
    print(f'{np.round(R, redondeo)}\t\t{np.round(S, redondeo)}\t\t{np.round(dp, redondeo)}\t\t{np.round(dq, redondeo)}')

def mostrarResultadosLU(L:np.array, U:np.array, d:np.array, x:np.array):
    """
    Funcion para mostrar las matrices L y U y los vectores d y x.
    """
    mostrarMatriz(L, 'L')
    mostrarMatriz(U, 'U')
    mostrarVector(d, 'd')
    mostrarVector(x, 'x')
    
def mostrarMatriz(matriz:np.array, nombre:str):
    """
    Funcion para mostrar una matriz.
    Parametros:
        matriz: matriz a mostrar.
        nombre: nombre de la matriz.
    """
    n = len(matriz)
    mitad = int(n / 2)
    len_nombre = len(nombre)
    espacio = ' ' * (len_nombre + 3)
    for i in range(n):
        if i == mitad:
            print(nombre + ' = ', end='')
        else:
            print(espacio, end='')
        for j in range(n):
            if matriz[i][j] >= 0:
                print(' ', end='')
            print(f'{np.around(matriz[i][j], decimals = 3)}   ', end='\t')
        print('')
    print('\n')

def mostrarVector(vector:np.array, nombre:str):
    """
    Funcion para mostrar un vector.
    Parametros:
        vector: vector a mostrar.
        nombre: nombre del vector.
    """
    n = len(vector)
    mitad = int(n / 2)
    len_nombre = len(nombre)
    espacio = ' ' * (len_nombre + 3)
    columna = np.around(vector, decimals = 3)
    for i in range(n):
        if i == mitad:
            print(nombre + ' = ', end='')
        else:
            print(espacio, end='')
        if vector[i] >= 0:
            print(' ', end='')
        print(f'{columna[i]}   ')
    print('\n')

def mostrarPolinomioCaracteristico(b:np.array):
    """
    Funcion para mostrar el polinomio caracteristico.
    Parametros:
        b: lista de coeficientes del polinomio caracteristico. -> [b_n-1, b_n-2, ..., b_0]
    """
    print('Polinomio caracteristico:')
    polinomio = ''
    grad = len(b) - 1
    for i, coeficiente in enumerate(b[:-2]):
        if coeficiente != 0:
            polinomio += f'{np.round(coeficiente, 3)}*λ**{grad - i} + '
    polinomio += f'{np.round(b[-2], 3)}*λ + '
    polinomio += f'{np.round(b[-1], 3)}'
    polinomio = polinomio.replace('+ -', '- ')
    x = sp.Symbol('λ')
    polinomio = sp.Poly(polinomio, x)
    display(Math(f'P_{grad}(λ) = ' + sp.latex(polinomio.as_expr())))

def mostrarValoresCaracteristicos(valores:np.array):
    """
    Funcion para mostrar los valores caracteristicos.
    Parametros:
        valores: lista de valores caracteristicos.
    """
    print('Valores caracteristicos:')
    for i, valor in enumerate(valores):
        if np.imag(valor) != 0:
            display(Math(f'λ_{i+1} = {np.round(float(valor), 3)}'))
        else:
            display(Math(f'λ_{i+1} = {np.round(valor, 3)}'))

# ------------------- Leer funciones, polinomios, constantes, etc. -------------------
def leerFuncion():
    """
    Funcion para leer un string y convertirlo en una funcion de sympy.
    """
    mensaje = 'Reglas para ingresar funciones:\n1.- La funcion debe de depender solo de la variable x\n2.- Para ingresar seno escriba sin(argumento)\n3.- Para ingresar el numero de Euler escriba exp(1)\n4.- Para ingresar el numero pi escriba pi'
    print(mensaje)
    funcion = input('Ingrese la función: ')
    x = sp.Symbol('x')
    try:
        funcion = sp.sympify(funcion)
        return funcion
    except:
        print('La función ingresada no es válida.')
        return None
    
def leerTolerancia()->float:
    """
    Función para validar la tolerancia.
    Retorna:
        tol: tolerancia.
    """
    mensaje = 'La tolerancia debe ser un número mayor que cero.'
    print(mensaje)
    tol = input('Ingrese la tolerancia: ')
    try:
        tol = float(tol)
    except:
        raise ValueError('El valor de la tolerancia debe ser un número.')
    if tol <= 0:
        raise ValueError('El valor de la tolerancia debe ser mayor que cero.')
    return tol

def leerPolinomio():
    """
    Funcion para leer los coeficientes de un polinomio.
    """
    mensaje = 'Reglas para ingresar polinomios:\n1.- EL grado del polinomio debe ser mayor a 0.\n2.- Los coeficientes deben ser números.\n3.- Se deben ingresar al menos dos coeficientes distintos de cero.'
    print(mensaje)
    grado = input('Ingrese el grado del polinomio: ')
    try:
        grado = int(grado)
    except:
        raise ValueError('El grado debe ser un número entero.')
    if grado <= 0:
        raise ValueError('El grado debe ser mayor a cero.')
    coeficientes = []
    contador = 0
    for i in range(grado + 1):
        coeficiente = input(f'Ingrese el coeficiente de grado {grado - i}: ')
        try:
            coeficiente = float(coeficiente)
        except:
            raise ValueError('El coeficiente debe ser un número.')
        if i == contador and coeficiente == 0:
            contador += 1
        else:
            coeficientes.append(coeficiente)
    if len(coeficientes) < 2:
        raise ValueError('Se deben ingresar al menos dos coeficientes distintos de cero.')
    return coeficientes

def leerLU():
    """
    Funcion para leer una matriz y un vector.
    Retorna:
        matriz: matriz leida.
        vector: vector leido.
    """
    print('Matriz de coeficientes:')
    matriz = leerMatriz()
    print('Vector de terminos independientes:')
    vector = leerVector(len(matriz))
    return matriz, vector

def leerMatriz()->np.array:
    """
    Funcion para leer una matriz de n x n.
    Retorna: 
        np.array: matriz leida.
    """
    mensaje = 'Reglas para ingresar una matriz:\n1.- La matriz debe ser cuadrada.\n2.- Los elementos deben ser números.'
    print(mensaje)
    n = input('Ingrese el tamaño de la matriz: ')
    try:
        n = int(n)
    except:
        raise ValueError('El tamaño de la matriz debe ser un número entero.')
    if n <= 0:
        raise ValueError('El tamaño de la matriz debe ser mayor a cero.')
    matriz = []
    for i in range(n):
        fila = []
        for j in range(n):
            elemento = input(f'Ingrese el elemento {i+1},{j+1}: ')
            try:
                elemento = float(elemento)
            except:
                raise ValueError('El elemento debe ser un número.')
            fila.append(elemento)
        matriz.append(fila)
    return np.array(matriz)

def leerVector(n:int, reglas='')->np.array:
    """
    Funcion para leer un vector de n elementos.
    Parametros:
        n: tamaño del vector.
        reglas: reglas adicionales para ingresar el vector.
    Retorna: 
        np.array: vector leido.
    """
    mensaje = 'Reglas para ingresar un vector:\n1.- Los elementos deben ser números.' + reglas
    print(mensaje)
    vector = []
    for i in range(n):
        elemento = input(f'Ingrese el elemento {i+1}: ')
        try:
            elemento = float(elemento)
        except:
            raise ValueError('El elemento debe ser un número.')
        vector.append(elemento)
    return np.array(vector)

def leerVectorKrilov(n:int)->np.array:
    """
    Funcion para leer un vector de n elementos.
    Parametros:
        n: tamaño del vector.
    Retorna: 
        np.array: vector leido.
    """
    mensaje = 'Si desea utilizar el vector de Krilov por defecto ingrese 0. En caso contrario ingrese 1.'
    print(mensaje)
    opcion = input('Ingrese la opción: ')
    if opcion == '0':
        vector = np.zeros(n) # Vector de ceros de tamaño n
        vector[0] = 1 # Primer elemento igual a 1
    else:
        vector = leerVector(n, '\n2.- Al menos un elemento debe ser distinto de cero.')
        # si todos los elementos son cero
        if np.all(vector == 0):
            print('Al menos un elemento debe ser distinto de cero.')
            vector = leerVectorKrilov(n)
    return vector

# ------------------- Validar datos -------------------
def validarDatosBiseccion(x_i:str, x_s:str, tol:str):
    """
    Función para validar los datos de entrada del método de bisección.
    Parámetros:
        x_i: valor inferior del intervalo.
        x_s: valor superior del intervalo.
        tol: tolerancia.
    Retorna:
        x_i: valor inicial del intervalo.
        x_s: valor final del intervalo.
        tol: tolerancia.
    """
    try:
        x_i = float(x_i)
    except:
        raise ValueError('El valor de x_i debe ser un número.')
    try:
        x_s = float(x_s)
    except:
        raise ValueError('El valor de x_s debe ser un número.')
    if x_i >= x_s:
        raise ValueError('El valor de x_i debe ser menor que el de x_s.')
    try:
        tol = float(tol)
    except:
        raise ValueError('El valor de la tolerancia debe ser un número.')
    if tol <= 0:
        raise ValueError('El valor de la tolerancia debe ser mayor que cero.')
    return x_i, x_s, tol

def validarDatosN_R(x_0:str, tol:str):
    """
    Función para validar los datos de entrada del método de Newton-Raphson.
    Parámetros:
        x_0: primera aproximacion de la raíz.
        tol: tolerancia.
    Retorna:
        x_0: primera aproximacion de la raíz.
        tol: tolerancia.
    """
    try:
        x_0 = float(x_0)
    except:
        raise ValueError('El valor de x_0 debe ser un número.')
    try:
        tol = float(tol)
    except:
        raise ValueError('El valor de la tolerancia debe ser un número.')
    if tol <= 0:
        raise ValueError('El valor de la tolerancia debe ser mayor que cero.')
    return x_0, tol

# ------------------- Funciones adicionales -------------------
def quitarNan(valores_x:list, valores_y:list) -> tuple:
    '''
    Funcion para quitar los valores nan de una lista de valores.
    Parametros:
        valores_x: lista de valores de x.
        valores_y: lista de valores de y.
    Retorna:
        valores_x: lista de valores de x modificada.
        valores_y: lista de valores de y sin Nan.
    '''
    valores_x_actualizado = []
    valores_y_actualizado = []
    len_y = len(valores_y)
    for i in range(len_y):
        if not np.isnan(valores_y[i]) and not np.isinf(valores_y[i]):
            valores_x_actualizado.append(valores_x[i])
            valores_y_actualizado.append(valores_y[i])
    return valores_x_actualizado, valores_y_actualizado


# funcion main para pruebas
if __name__ == '__main__':
    pass