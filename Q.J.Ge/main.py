import numpy as np
import cmath
from math import cos, sin, atan



def path_gen(L, th1, phi, r, alpha, n, x0, y0):

    th2 = phi + th1
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

    px1 = L[1]*cos(phi+th1) + r*cos(alpha+th3_1) + x0
    px2 = L[1]*cos(phi+th1) + r*cos(alpha+th3_2) + x0
    py1 = L[1]*sin(phi+th1) + r*sin(alpha+th3_1) + y0
    py2 = L[1]*sin(phi+th1) + r*sin(alpha+th3_2) + y0
    pass

def loop_equation(L, th1, phi, r, alpha, n, x0, y0):
    A = L[2]*(cmath.rect(L[1],-phi) - L[0])
    B = sum(L*L) - 2*L[0]*L[1]*cos(phi)
    C = L[2]*(cmath.rect(L[1],phi) - L[0])

    delta_1 = L[0]**2 + L[1]**2 - (L[2] + L[3])**2 - 2*L[0]*L[1]*cos(phi)
    delta_2 = L[0]**2 + L[1]**2 - (L[2] - L[3])**2 - 2*L[0]*L[1]*cos(phi)

    pass
