import numpy as np

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