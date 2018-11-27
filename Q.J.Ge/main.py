import numpy as np
import cmath
import matplotlib.pyplot as plt
from math import cos, sin, atan, pi, sqrt

def path_gen(L, th1, phi_0, r, alpha, n, x0, y0):

    p1 = []
    p2 = []
    th2 = phi_0 + th1
    for i in range(n):
        th2 = th2 + 2*pi/n
        k1 = L[0]**2 + L[1]**2 + L[2]**2 - L[3]**2 - 2*L[0]*L[1]*cos(th2-th1)
        k2 = 2*L[0]*L[2]*cos(th1) - 2*L[1]*L[2]*cos(th2)
        k3 = 2*L[0]*L[2]*sin(th1) - 2*L[1]*L[2]*sin(th2)
        a = k1 + k2
        b = -2 * k3
        c = k1 -k2
        x_1 = (-b + sqrt(b**2 - 4*a*c)) / (2 * a)
        x_2 = (-b - sqrt(b**2 - 4*a*c)) / (2 * a)

        th3_1 = 2*atan(x_1)
        th3_2 = 2*atan(x_2)

        temp = []
        p1x = L[1]*cos(th2) + r*cos(alpha+th3_1) + x0
        p1y = L[1]*sin(th2) + r*sin(alpha+th3_1) + y0
        temp.append(p1x)
        temp.append(p1y)
        p1.append(temp)

        temp = []
        p2x = L[1]*cos(th2) + r*cos(alpha+th3_2) + x0
        p2y = L[1]*sin(th2) + r*sin(alpha+th3_2) + y0
        temp.append(p2x)
        temp.append(p2y)
        p2.append(temp)

    p1 = np.array(p1)
    p2 = np.array(p2)

    point = p1
    plt.plot(point[:,0],point[:,1],'ro')
    plt.axis('equal')
    plt.show()

    return p1, p2

def loop_equation(L, th1, phi_0, r, alpha, n, x0, y0):
    A = L[2]*(cmath.rect(L[1],-phi_0) - L[0])
    B = sum(L*L) - 2*L[0]*L[1]*cos(phi_0)
    C = L[2]*(cmath.rect(L[1],phi_0) - L[0])

    delta_1 = L[0]**2 + L[1]**2 - (L[2] + L[3])**2 - 2*L[0]*L[1]*cos(phi_0)
    delta_2 = L[0]**2 + L[1]**2 - (L[2] - L[3])**2 - 2*L[0]*L[1]*cos(phi_0)

    pass

if __name__ == "__main__":
    # Test data No.1
    L = np.array([40, 20, 40, 30])
    th1 = 0
    phi_0 = pi/2
    r = 20*sqrt(2)
    alpha = pi/4
    n = 360
    x0 = 0
    y0 = 0
    #=======================================================
    # Path generation
    p1, p2 = path_gen(L, th1, phi_0, r, alpha, n, x0, y0)
