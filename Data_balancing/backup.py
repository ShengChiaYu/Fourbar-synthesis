def curve_normalization(p1, fn, plot=False):
    ## Step 1: evaluate the length L of the curve
    p1_tmp = np.concatenate((p1[1:], p1[0].reshape(1,-1)), axis=0)
    Li = np.sqrt(np.sum((p1-p1_tmp)**2, axis=1))
    # print(Li)

    ## Step 2: determine the location of the centre of gravity (c.g.) of the curve.
    cx = np.sum((p1+p1_tmp)[:,0]*Li) / 2 / np.sum(Li)
    cy = np.sum((p1+p1_tmp)[:,1]*Li) / 2 / np.sum(Li)
    print("cx: {:.3f}, cy: {:.3f}".format(cx,cy))

    ## Step 3: evaluate the moments of inertia Ixx , Iyy and Ixy of the polygon with respect to its c.g.
    xi1 = p1[:,0]
    yi1 = p1[:,1]
    xi = p1_tmp[:,0]
    yi = p1_tmp[:,1]
    Ixx = np.sum(Li*( (yi1-cy)**2 + (yi-cy)**2 + (yi1-cy)*(yi-cy) )) / 3
    Iyy = np.sum(Li*( (xi1-cx)**2 + (xi-cx)**2 + (xi1-cx)*(xi-cx) )) / 3
    Ixy = np.sum(Li*( (xi1-cx)*(yi-cy)+(xi-cx)*(yi1-cy) ) + 2*( (xi1-cx)*(yi1-cy)+(xi-cx)*(yi-cy) )) / 6

    ## Step 4: determine the direction, a, of the major principal axis with respect to the x-axis.
    if Ixx < Iyy:
        alpha = 0.5 * atan(2*Ixy / (Iyy - Ixx))
    elif Ixx > Iyy:
        alpha = 0.5 * atan(2*Ixy / (Iyy - Ixx)) + pi/2
    else:
        alpha = 0
    print("Ixx: {:.3f}".format(Ixx))
    print("Iyy: {:.3f}".format(Iyy))
    print("Ixy: {:.3f}".format(Ixy))
    print("alpha: {:.3f} deg".format(degrees(alpha)))

    ## Step 5: rotate the polygon by an angle alpha
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array(((c, s), (-s, c)))
    # p1c = p1 - np.array([cx, cy])
    # p1_rot = np.matmul(R,p1c.T).T + np.array([cx, cy])
    p1_rot = np.matmul(R,p1.T).T

    ## Step 6: evaluate the width w of the bounding box of the resulting polygon.
    Px = np.min(p1_rot[:,0])
    Py = np.min(p1_rot[:,1])
    Qx = np.max(p1_rot[:,0])
    Qy = np.max(p1_rot[:,1])
    w = Qx - Px
    h = Qy - Py
    # print(w, h)

    ## Step 7: bring the polygon to its normalized configuration.
    # p1_norm = p1_rot - np.array([Px, Py])
    # p1_norm = p1_norm / w
    T = np.array(((c/w, s/w, -Px/w), (-s/w, c/w, -Py/w), (0, 0, 1)))
    p1 = np.concatenate((p1, np.ones((p1.shape[0],1))), axis=1)
    p1_norm = np.matmul(T,p1.T)[:2,:].T
    # print(p1_norm)

    ## Step 8: taking the starting point of the path
    r = LA.norm(p1_norm, axis=1)
    ind = np.argmin(r)
    str_p = p1_norm[ind,:]
    # print(r)
    # print(str_p)

    ## Step 9: plot the normalized results
    if plot:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(p1[:,0],p1[:,1],label='original')
        # ax.plot(p1_rot[:,0],p1_rot[:,1],label='rotated')
        ax.plot(p1_norm[:,0],p1_norm[:,1],label='normalized')
        ax.plot(cx,cy,'b+',label='c.g.')
        ax.plot(str_p[0],str_p[1],'r+',label='s.p.')
        ax.legend(loc='upper left')
        ax.set_title('{} {}'.format(fn, degrees(alpha)))
        plt.axis('equal')
        plt.savefig('./test_figure/curve_normalization/{} {:.3f}.png'.format(fn, degrees(alpha)))


    return p1_norm, str_p


def curve_normalization_pca(p1, fn, plot=False):

    ## Step 1: calculate the mean and std of path points
    x_mean, y_mean = np.mean(p1, axis=0)
    x_std, y_std = np.std(p1, axis=0)

    ## Step 2: calculate the prinpal component axes (eigenvectors)
    m = p1.shape[0]
    Cxx = np.sum((p1[:,0]-x_mean)**2) / m
    Cyy = np.sum((p1[:,1]-y_mean)**2) / m
    Cxy = np.sum((p1[:,0]-x_mean)*(p1[:,1]-y_mean)) / m
    C = np.array(((Cxx,Cxy),(Cxy,Cyy)))
    w, v = LA.eig(C)
    print(w)
    print(v)
    # print(v[:,0].dot(v[:,1]))

    theta_1 = atan2(v[1][0],v[0][0])
    theta_2 = atan2(v[1][1],v[0][1])
    pca = np.insert(v.T, 1, (0,0), axis=0) + np.mean(p1, axis=0)
    # print(degrees(theta_1))
    # print(degrees(theta_2))


    ## Step 3: calculate the rotation matrix
    if theta_1*theta_2 > 0:
        alpha = theta_1 if theta_1 < theta_2 else theta_2
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array(((c, s), (-s, c)))

    elif theta_1*theta_2 < 0:
        if abs(theta_1) < pi/2:
            alpha = theta_1 if theta_1 < theta_2 else theta_2
        else:
            alpha = theta_1 if theta_1 > theta_2 else theta_2
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array(((c, s), (-s, c)))
    else:
        if theta_1 == -pi/2 or theta_2 == -pi/2:
            alpha = pi/2
            c, s = np.cos(alpha), np.sin(alpha)
            R = np.array(((c, -s), (s, c)))

    ## Step 4: normalized the path points
    p1_normal = (p1 - np.array([x_mean, y_mean])) / np.array([x_std, y_std])
    p1_normal = np.matmul(R,p1_normal.T).T

    ## Step 5: taking the starting point of the path
    r = LA.norm(p1_normal, axis=1)
    ind = np.argmin(r)
    str_p = p1_normal[ind,:]

    if plot:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(p1[:,0],p1[:,1],label='original')
        ax.plot(p1_normal[:,0],p1_normal[:,1],label='normalized')
        ax.plot(pca[:,0], pca[:,1], color='blue', linestyle='dashed',linewidth=2,label='pca')
        ax.plot(str_p[0],str_p[1],'r+',label='s.p.')
        ax.legend(loc='upper left')
        ax.set_title('{} {}'.format(fn, degrees(alpha)))
        plt.axis('equal')
        plt.savefig('./test_figure/curve_normalization_pca/{} {:.3f}.png'.format(fn, degrees(alpha)))

    return p1_normal, str_p
