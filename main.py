import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from scipy.optimize import minimize
from scipy import integrate

camera_num = 100
domain = [[0, 0], [1, 0], [1, 1], [0, 1]]
extra = [[-1, -1], [2, -1], [2, 2], [-1, 2]]
viewport = [[0, 1], [0, 1]]

r = np.ones((camera_num, 1))
for i in range(camera_num):
    r[i] = (1 + i % camera_num) * 0.1


cells = [0] * camera_num


def rho(x):
    return math.exp(- 10 * (x[0] * x[0] + x[1] * x[1]))


def integrant_m(s, t, r1, r2, r3):
    r_st = r1 + (r2-r1)*s + (r3-r1)*t
    return rho(r_st)


def integrant(s, t, r1, r2, r3, xi):
    r_st = r1 + (r2-r1)*s + (r3-r1)*t
    d = np.linalg.norm(r_st - xi)
    return rho(r_st) * d * d


def F(X):
    global domain
    pts = np.reshape(X, (camera_num, 2))
    pts = np.vstack((pts, extra))
    vor = Voronoi(pts)
    ret = 0.0
    for i in range(camera_num):
        poly_verts = [vor.vertices[v]
                      for v in vor.regions[vor.point_region[i]]]
        cells[i] = Polygon(domain).intersection(
            Polygon(poly_verts))  # trim cell by domain
        if not cells[i].centroid.coords:
            continue
        centroid = np.array(cells[i].centroid.coords[0])
        verts = np.array(cells[i].exterior.coords)

        triangles = [
            [centroid, verts[j], verts[(j+1) % len(verts)]] for j in range(len(verts))]
        for triangle in triangles:
            arg = (triangle[0], triangle[1], triangle[2], tuple(pts[i]))
            s = Polygon(triangle).area
            I2 = integrate.dblquad(
                integrant, 0, 1, lambda s: 0, lambda s: 1-s, args=arg)
            ret += s * I2[0]
    return ret


def G(X):
    ret = np.zeros((camera_num, 2))
    pts = np.reshape(X, (camera_num, 2))
    pts = np.vstack((pts, extra))
    for i in range(camera_num):
        if not cells[i].centroid.coords:
            continue
        centroid = np.array(cells[i].centroid.coords[0])
        verts = np.array(cells[i].exterior.coords)

        triangles = [
            [centroid, verts[j], verts[(j+1) % len(verts)]] for j in range(len(verts))]
        m_i = 0
        for triangle in triangles:
            arg = (triangle[0], triangle[1], triangle[2])
            s = Polygon(triangle).area
            I2 = integrate.dblquad(
                integrant_m, 0, 1, lambda s: 0, lambda s: 1-s, args=arg)
            m_i += s * I2[0]

        ret[i] = 2 * (pts[i] - centroid) * m_i
    return ret.flatten()


count = 0


def cbf(X):
    global count
    plt.figure(figsize=(6, 6))
    plt.cla()

    pts = np.reshape(X, (camera_num, 2))
    pts = np.vstack((pts, extra))
    ax = plt.gcf().gca()
    voronoi_plot_2d(Voronoi(pts), ax=ax, show_vertices=False)

    # for i in range(camera_num):
    #     circle = plt.Circle(pts[i], r[i], alpha=0.1, color='blue')
    #     ax.add_artist(circle)

    plt.gca().set_aspect('equal')
    plt.gca().set_xlim(viewport[0])
    plt.gca().set_ylim(viewport[1])
    plt.savefig('output/' + str(count).zfill(2) + '.png', bbox_inches='tight')
    # plt.show()
    count += 1


if __name__ == '__main__':
    random.seed(1224)
    X0 = [[0.1*random.uniform(0.0, 1.0), 0.1*random.uniform(0.0, 1.0)]
          for i in range(camera_num)]

    cbf(X0)

    minimize(F, X0, method="L-BFGS-B", jac=G, callback=cbf,
             options={'maxiter': 100})
    print("iteration: ", count)
