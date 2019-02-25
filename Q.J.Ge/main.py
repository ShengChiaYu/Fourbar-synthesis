import numpy as np
import matplotlib.pyplot as plt
import math
from cmath import sqrt
from math import cos, sin, atan, acos, pi, degrees

def path_gen_open(L, th1, r, alpha, n, x0, y0):

    # Validity: Check if the linkages can be assembled and move.
    if max(L) >= (sum(L) - max(L)):
        print("Impossible geometry.")
        return 0, 0

    # Limit of rotation angle of input linkage
    condition_1 = (L[0] + L[1]) - (L[2] + L[3])
    condition_2 = abs(L[0] - L[1]) - abs(L[2] - L[3])

    # Upper limit exists
    if condition_1 > 0  and condition_2 >= 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(-th2_max, th2_max, n)
        plt.title("Upper limit exists.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(-th2_max.real), degrees(th2_max.real)))
        print("Upper limit exists.")
    # Lower limit exists
    elif condition_1 <= 0 and condition_2 < 0:
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, 2*pi - th2_min, n)
        plt.title("Lower limit exists.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(2*pi-th2_min.real)))
        print("Lower limit exists.")
    # Both limit exist
    elif condition_1 > 0 and condition_2 < 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, th2_max, n)
        #th2 = np.linspace(-th2_max, -th2_min, n)
        plt.title("Both limit exist.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(th2_max.real)))
        print("Both limit exist.")
    # No limit exists
    elif condition_1 <= 0 and condition_2 >= 0:
        th2 = np.linspace(0, 2*pi, n)
        plt.title("No limit exists.")
        plt.xlabel("Th_2 min = 0, Th_2 max = 360")
        print("No limit exists.")

    # Calculate the positions of coupler curve by different input angles
    p1 = []
    p2 = []
    for i in range(n):
        k1 = L[0]**2 + L[1]**2 + L[2]**2 - L[3]**2 - 2*L[0]*L[1]*cos(th2[i]-th1)
        k2 = 2*L[0]*L[2]*cos(th1) - 2*L[1]*L[2]*cos(th2[i])
        k3 = 2*L[0]*L[2]*sin(th1) - 2*L[1]*L[2]*sin(th2[i])
        a = k1 + k2
        b = -2 * k3
        c = k1 -k2

        x_1 = (-b + sqrt(b**2 - 4*a*c).real) / (2 * a) # x_1 and x_2 = tan((1/2)*th3)
        x_2 = (-b - sqrt(b**2 - 4*a*c).real) / (2 * a)

        th3_1 = 2*atan(x_1)
        th3_2 = 2*atan(x_2)

        p1x = L[1]*cos(th2[i]) + r*cos(alpha+th3_1) + x0
        p1y = L[1]*sin(th2[i]) + r*sin(alpha+th3_1) + y0
        p1.append([p1x, p1y])

        p2x = L[1]*cos(th2[i]) + r*cos(alpha+th3_2) + x0
        p2y = L[1]*sin(th2[i]) + r*sin(alpha+th3_2) + y0
        p2.append([p2x, p2y])

        plt.plot(L[1]*cos(th2[i]) + x0, L[1]*sin(th2[i]) + y0, 'go', markersize=1)

    p1 = np.array(p1)
    p2 = np.array(p2)

    plt.plot(p1[:,0],p1[:,1],'ro', markersize=1)
    plt.plot(p2[:,0],p2[:,1],'bo', markersize=1)
    plt.plot(x0,y0,'k+')
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

    # Testing data
    #L = np.array([30, 40, 31, 38]) # Upper limit exists. c1>0  c2>=0
    #L = np.array([20, 29, 30, 40]) # Lower limit exists. c1<=0 c2<0
    #L = np.array([31, 40, 20, 40]) # Both limit exist.   c1>0  c2<0
    L = np.array([20, 40, 30, 40]) # No limit exists.    c1<=0 c2>=0
    th1 = 0
    r = L[2]/2*math.sqrt(2) # 50 % of coupler length
    alpha = pi/4       # midpoint of coupler link
    n = 360
    x0 = 0
    y0 = 0

    """
    # Q.J.Ge Closed path parameters
    L = np.array([11, 6, 8, 10])
    th1 = 0.1745
    r = 7
    alpha = 0.6981
    n = 360
    x0 = 10
    y0 = 14
    """
    """
    # Q.J.Ge open path parameters
    L = np.array([3, 1, 2, 1.6])
    th1 = 0.2
    r = 0.5
    alpha = 0.3
    n = 360
    x0 = -2
    y0 = -3
    """
    #=======================================================
    # Path generation
    p1, p2 = path_gen_open(L, th1, r, alpha, n, x0, y0)
    print(p1)
