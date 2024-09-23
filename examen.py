import numpy as np
import math

# Métodos de optimización simplex, gran M, dos fases
def metodo_simplex():
    print("Ejecutando el Método Simplex...")
    
    # Pedir los datos del usuario
    c, A, b = pedir_datos_simplex()

    # Llamada al método simplex
    solucion = simplex(c, A, b)
    print("Solución óptima:", solucion)
    print("--------------------------------------")
    input("Presiona 'enter' para regresar al menu")


def metodo_gran_m():
    print("Ejecutando el Método de la Gran M...")
    
    # Pedir los datos del usuario
    c, A, b, signos = pedir_datos_gran_m()

    # Llamada al método de la Gran M
    solucion = gran_m(c, A, b, signos)
    print("Solución óptima:", solucion)
    print("--------------------------------------")
    input("Presiona 'enter' para regresar al menu")

def metodo_dos_fases():
    print("Ejecutando el Método de las Dos Fases...")
    
    print("--------------------------------------")
    input("Presiona 'enter' para regresar al menu")

# ----------------- Implementación del Método Simplex ---------------------
def simplex(c, A, b):
    tableau = convertir_a_tableau(c, A, b)

    while se_puede_mejorar(tableau):
        posicion_pivote = obtener_posicion_pivote(tableau)
        tableau = paso_pivote(tableau, posicion_pivote)

    return obtener_solucion(tableau)

# Convertir el problema a formato de tableau
def convertir_a_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]

# Verificar si la función objetivo se puede mejorar
def se_puede_mejorar(tableau):
    z = tableau[-1]
    return any(x > 0 for x in z[:-1])

# Obtener la posición del pivote
def obtener_posicion_pivote(tableau):
    z = tableau[-1]
    columna = next(i for i, x in enumerate(z[:-1]) if x > 0)

    restricciones = []
    for eq in tableau[:-1]:
        el = eq[columna]
        restricciones.append(math.inf if el <= 0 else eq[-1] / el)

    fila = restricciones.index(min(restricciones))
    return fila, columna

# Realizar el paso de pivote
def paso_pivote(tableau, posicion_pivote):
    nuevo_tableau = [[] for eq in tableau]

    i, j = posicion_pivote
    valor_pivote = tableau[i][j]
    nuevo_tableau[i] = np.array(tableau[i]) / valor_pivote

    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplicador = np.array(nuevo_tableau[i]) * tableau[eq_i][j]
            nuevo_tableau[eq_i] = np.array(tableau[eq_i]) - multiplicador

    return nuevo_tableau

# Obtener la solución del problema desde el tableau
def es_basico(columna):
    return sum(columna) == 1 and len([c for c in columna if c == 0]) == len(columna) - 1

def obtener_solucion(tableau):
    columnas = np.array(tableau).T
    soluciones = []
    for columna in columnas[:-1]:
        solucion = 0
        if es_basico(columna):
            indice_uno = columna.tolist().index(1)
            solucion = columnas[-1][indice_uno]
        soluciones.append(solucion)

    return soluciones

# Pedir los datos del problema al usuario para el método Simplex
def pedir_datos_simplex():
    # Pedir los coeficientes de la función objetivo
    c = list(map(float, input("Introduce los coeficientes de la función objetivo separados por espacios: ").split()))

    # Pedir la cantidad de restricciones
    num_restricciones = int(input("Introduce la cantidad de restricciones: "))

    A = []
    b = []

    # Pedir las restricciones
    for i in range(num_restricciones):
        restriccion = list(map(float, input(f"Introduce los coeficientes de la restricción {i+1} separados por espacios: ").split()))
        A.append(restriccion)
        termino_independiente = float(input(f"Introduce el término independiente de la restricción {i+1}: "))
        b.append(termino_independiente)

    return c, A, b

# ----------------- Implementación del Método de la Gran M ---------------------
def pedir_datos_gran_m():
    # Pedir los coeficientes de la función objetivo
    c = list(map(float, input("Introduce los coeficientes de la función objetivo separados por espacios: ").split()))

    # Preguntar si se va a maximizar o minimizar
    objetivo = input("¿Deseas maximizar o minimizar la función objetivo? (max/min): ").strip().lower()
    if objetivo == 'max':
        c = [-ci for ci in c]  # Convertimos a minimización multiplicando por -1
    elif objetivo != 'min':
        print("Opción no válida. Se asumirá minimización.")

    # Pedir la cantidad de restricciones
    num_restricciones = int(input("Introduce la cantidad de restricciones: "))

    A = []
    b = []
    signos = []

    # Pedir las restricciones
    for i in range(num_restricciones):
        restriccion = list(map(float, input(f"Introduce los coeficientes de la restricción {i+1} separados por espacios: ").split()))
        signo = input(f"Introduce el signo de la restricción {i+1} (<=, >=, =): ").strip()
        termino_independiente = float(input(f"Introduce el término independiente de la restricción {i+1}: "))

        A.append(restriccion)
        signos.append(signo)
        b.append(termino_independiente)

    return c, A, b, signos

def gran_m(c, A, b, signos):
    M = 1e5  # Valor grande para M
    num_vars = len(c)
    num_restricciones = len(A)

    # Inicializar matrices
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # Listas para variables de holgura y artificiales
    vars_holgura = []
    vars_artificiales = []

    # Construir A modificada, c modificada
    for i in range(num_restricciones):
        if signos[i] == '<=':
            # Añadir variable de holgura
            col_holgura = np.zeros((num_restricciones, 1))
            col_holgura[i][0] = 1
            A = np.hstack((A, col_holgura))
            c = np.append(c, 0)
            vars_holgura.append(num_vars)
            num_vars += 1
        elif signos[i] == '>=':
            # Añadir variable de holgura negativa y variable artificial
            col_holgura = np.zeros((num_restricciones, 1))
            col_holgura[i][0] = -1
            A = np.hstack((A, col_holgura))
            c = np.append(c, 0)
            vars_holgura.append(num_vars)
            num_vars += 1

            # Añadir variable artificial
            col_artificial = np.zeros((num_restricciones, 1))
            col_artificial[i][0] = 1
            A = np.hstack((A, col_artificial))
            c = np.append(c, M)
            vars_artificiales.append(num_vars)
            num_vars += 1
        elif signos[i] == '=':
            # Añadir variable artificial
            col_artificial = np.zeros((num_restricciones, 1))
            col_artificial[i][0] = 1
            A = np.hstack((A, col_artificial))
            c = np.append(c, M)
            vars_artificiales.append(num_vars)
            num_vars += 1
        else:
            print(f"Signo '{signos[i]}' no reconocido en la restricción {i+1}.")
            return None

    # Construir tableau inicial
    tableau = np.hstack((A, b.reshape(-1,1)))
    c = np.append(c, 0)  # Agregar cero para el término independiente
    tableau = np.vstack((tableau, c))

    # Aplicar método Simplex
    solucion = simplex_gran_m(tableau, vars_artificiales)

    return solucion

def simplex_gran_m(tableau, vars_artificiales):
    num_filas, num_columnas = tableau.shape

    while True:
        # Identificar columna pivote (menor valor en la fila de costos, excluyendo el último elemento)
        costos = tableau[-1, :-1]
        columna_pivote = np.argmin(costos)
        valor_minimo = costos[columna_pivote]

        # Si todos los costos son mayores o iguales a cero, se ha alcanzado la solución óptima
        if valor_minimo >= 0:
            break

        # Identificar fila pivote
        restricciones = tableau[:-1, columna_pivote]
        rhs = tableau[:-1, -1]
        cocientes = []

        for i in range(len(restricciones)):
            if restricciones[i] > 0:
                cocientes.append(rhs[i] / restricciones[i])
            else:
                cocientes.append(np.inf)

        fila_pivote = np.argmin(cocientes)

        # Verificar si el problema no tiene solución óptima
        if all(c == np.inf for c in cocientes):
            print("El problema no tiene solución óptima.")
            return None

        # Realizar pivote
        pivote = tableau[fila_pivote, columna_pivote]
        tableau[fila_pivote, :] = tableau[fila_pivote, :] / pivote

        for i in range(num_filas):
            if i != fila_pivote:
                factor = tableau[i, columna_pivote]
                tableau[i, :] = tableau[i, :] - factor * tableau[fila_pivote, :]

    # Obtener solución
    variables = tableau[:-1, :-1]
    rhs = tableau[:-1, -1]
    num_vars = variables.shape[1]
    solucion = np.zeros(num_vars)

    for i in range(num_vars):
        columna = variables[:, i]
        if np.count_nonzero(columna) == 1:
            fila = np.where(columna == 1)[0][0]
            solucion[i] = rhs[fila]

    # Verificar si alguna variable artificial está en la solución básica
    for var in vars_artificiales:
        if solucion[var] != 0:
            print("El problema no tiene solución factible.")
            return None

    valor_funcion_objetivo = tableau[-1, -1]
    print(f"Valor óptimo de la función objetivo: {valor_funcion_objetivo}")

    return solucion[:num_vars - len(vars_artificiales)]  # Excluir variables artificiales

# ------------------------------------------------------------------------------


# Menú interactivo
def mostrar_menu():
    print("\nSelecciona el metodo que deseas utilizar:")
    print("1. Metodo Simplex")
    print("2. Metodo de la Gran M")
    print("3. Metodo de las Dos Fases")
    print("4. Salir")

def ejecutar_seleccion(opcion):
    if opcion == "1":
        metodo_simplex()
    elif opcion == "2":
        metodo_gran_m()
    elif opcion == "3":
        metodo_dos_fases()
    elif opcion == "4":
        print("Saliendo del programa.")
    else:
        print("Opción no válida. Intenta nuevamente.")

# Función principal para ejecutar el menú
def main():
    opcion = ""
    while opcion != "4":
        mostrar_menu()
        opcion = input("Elige una opción: ")
        ejecutar_seleccion(opcion)

if __name__ == "__main__":
    main()

