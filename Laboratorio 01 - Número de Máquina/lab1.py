import math
import numpy as np
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

ejercicio_04()