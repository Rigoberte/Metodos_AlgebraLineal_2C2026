import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def rota(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R

def escala(s):
    n = len(s)
    S = np.zeros((n, n))

    for i in range(n):
        S[i, i] = s[i]

    return S

def rota_y_escala(theta, s):
    return escala(s) @ rota(theta)

def afin(theta, s, b):

    rot_y_esc = rota_y_escala(theta, s)

    A = np.zeros((3, 3))

    A[0:2, 0:2] = rot_y_esc
    A[0:2, 2] = b
    A[2, 2] = 1

    return A

def trans_afin(v, theta, s, b):
    A = afin(theta, s, b)

    v_hom = np.array([v[0], v[1], 1])

    v_trans = A @ v_hom

    return v_trans[0:2]

def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
    
    return wz

def proyectarPts(T, wz):
    """
    Retornar los puntos transformados xy = (x/2, y/2) usando la matriz de transformacion T
    """
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = None
    ############### Insert code here!! ######################3    
    xy = np.zeros(np.shape(wz))

    xy[0, :] = T[0, 0] * wz[0, :] + T[0, 1] * wz[1, :]
    xy[1, :] = T[1, 0] * wz[0, :] + T[1, 1] * wz[1, :]
    ############### Insert code here!! ######################3
    return xy


def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
            [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')    
    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
        xlim=limits[0], ylim=limits[1],
        xlabel=a_label, ylabel=b_label)

def ejercicio01():
    # generar el tipo de transformacion dando valores a la matriz T
    csv_path = 'Recursos - Laboratorio 02 - Transformaciones Lineales-20260825/T.csv'
    
    T = pd.read_csv(csv_path, header=None).values
    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')

def ejercicio02():
    a = 2
    b = 3

    T = np.array([
        [a, 0],
        [0, b]
    ])

    corners = np.array([
        [0, 0],
        [100, 100]
    ])

    wz = pointsGrid(corners)

    vistform(T, wz, 'Reescalamiento')

def ejercicio03():
    c = 0
    d = 1

    if c*d == 1:
        print('La matriz no es invertible')
        return

    T = np.array([
        [1, c],
        [d, 1]
    ])

    T_inv = 1/(1-d*c) * np.array([
        [1, -c],
        [-d, 1]
    ])

    corners = np.array([
        [0, 0],
        [100, 100]
    ])

    wz = pointsGrid(corners)

    vistform(T, wz, 'Transformacion lineal')
    vistform(T_inv, wz, 'Inversa de la transformacion lineal')

def ejercicio04():
    theta = np.pi
    R = rota(theta)

    corners = np.array([
        [0, 0],
        [100, 100]
    ])

    wz = pointsGrid(corners)

    vistform(R, wz, 'Rotacion de coordenadas')

def ejercicio05():
    """ Rotar 45, escalar 2 en x y 3 en y, rotar -45 """

    theta = np.pi/4
    s = np.array([2, 3])

    R1 = rota(theta)
    S = np.array([
        [s[0], 0],
        [0, s[1]]
    ])
    R2 = rota(-theta)

    T = R2 @ S @ R1
    

    corners = np.array([
        [0, 0],
        [100, 100]
    ])

    wz = pointsGrid(corners)

    vistform(T, wz, 'Rotar 45, escalar 2 en x y 3 en y, rotar -45')

def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    #ejercicio01()
    #ejercicio02()
    #ejercicio03()
    #ejercicio04()
    ejercicio05()

    while True:
        try:
            plt.pause(0.1)
        except KeyboardInterrupt:
            print('Saliendo del programa')
            break
    
    
if __name__ == "__main__":
    main()
