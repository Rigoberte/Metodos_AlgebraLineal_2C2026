import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# Ejercicio 01
def ejercicio_01():
    print("0.1 en base 10:")
    print(format(0.1, '.20f'))  # Muestra el número con 20 decimales
    print (0.1 + 0.1 + 0.1 == 0.3) 

def ejercicio_02():
    a=1.0
    while a!= 0.1:
        print(a)
        #a=a-0.1
        a = np.round(a - 0.1, 10)  # Redondea a 10 decimales para evitar errores de precisión
    print('fin')

def ejercicio_03():
    print("\n0.3 + 0.25")
    print(0.3 + 0.25)

    print("\n0.3 - 0.25")
    print(0.3 - 0.25)

    print("\n0.25 en base 2")
    m, e = math.frexp(0.25)
    print("Mantisa:", m)
    print("Exponente:", e)

    print("\n0.3 en base 2")
    m, e = math.frexp(0.3)
    print("Mantisa:", m)
    print("Exponente:", e)

def ejercicio_04():
    print("\nnp.sqrt(2)**2-2 = ", np.sqrt(2)**2-2)

    range_x = np.linspace(0, 5e-8, 100)

    range_y = np.sqrt(2*range_x**2 + 1) - 1

    range_y_alt = (2*range_x**2) / (np.sqrt(2*range_x**2 + 1) + 1)

    plt.plot(range_x, range_y, label='Original')
    plt.plot(range_x, range_y_alt, label='Alternativa')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparación de funciones')
    plt.legend()
    plt.grid()
    plt.show()

def ejercicio_05():
    x = math.sqrt(2)

    l = []

    for i in range(100):
        l.append(x)
        x = x * x / math.sqrt(2)

    print(l)

    plt.plot(l)
    plt.axhline(math.sqrt(2), color="red")
    plt.show()

def ejercicio_06(func):
    n = 7
    s = func(0)
    for i in range(1,10**n+1):
        s = s + func(1/i)
    print('suma = ', s)
    
    s = func(0)
    for i in range(1,5*10**n+1):
        s = s + func(1/i)
    print('suma = ', s)

    s = func(0)
    for i in range(2*10**n,0,-1):
        s = s + func(1/i)
    print('suma = ', s)

def ejercicio_06_extra():
    n = 10
    s = np.float64(0)
    for i in range(0,n+1):
        s = s + np.float64(1/math.factorial(i))
    print('Euler estimado = ', s)

    e = np.float64(np.e)
    print('Euler real = ', e)

    print('Diferencia = ', e - s)

#ejercicio_06(np.float32)
#ejercicio_06(np.float64)
#ejercicio_06_extra()

def matricesIguales(A, B):
    if A.shape != B.shape:
        return False

    if A.dtype != B.dtype:
        return False
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = A[i, j]
            b = B[i, j]
            if np.round(a, 10) != np.round(b, 10):
                return False
    return True

ej7_A = np.array([[4., 2., 1.], [2., 7., 9.], [0., 5., 22/3]])
ej7_L = np.array([[1., 0., 0.], [0.5, 1., 0.], [0., 5/6, 1.]])
ej7_U = np.array([[4., 2., 1.], [0., 6., 8.5], [0., 0., 0.25]])

#ejercicio_07 = print(matricesIguales(ej7_L @ ej7_U, ej7_A))

def esCuadrada(A):
    return A.shape[0] == A.shape[1]

def esSimetrica(A):
    if not esCuadrada(A):
        return False
    
    n = len(A)

    for i in range(n):
        for j in range (n):
            if A[i][j] != A[j][i]:
                return False
    
    d = A[0][0]
    for i in range(n):
        if A[i][i] != d:
            return False

    return True

def ejercicio08():
    A = np.array(np.random.rand(4, 4))

    print("A =", A)
    print("A.T@A =", A.T@A)

    print("esSimetrica(A.T@A) =", esSimetrica(A.T@A))

    print("esSimetrica(A.T@((A*0.25)/0.25)) =", esSimetrica(A.T@((A*0.25)/0.25)))

    print("esSimetrica(A.T@((A*0.2)/0.2)) =", esSimetrica(A.T@((A*0.2)/0.2)))

ejercicio08()