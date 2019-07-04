import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize


camera_num = 30
domain = [[0, 0], [1, 0], [1, 1], [0, 1]]

r = np.ones((camera_num, 1))
for i in range(camera_num):
    # r[i] = (1 + i % 3) * 0.1
    r[i] = 0


S = np.zeros((camera_num, 1))
C = np.zeros((camera_num, 2))


def F(X):
    global domain
    pts = np.reshape(X, (camera_num, 2))
    pts = np.vstack((pts, domain))
    vor = Voronoi(pts)
    ret = 0.0
    for i in range(camera_num):
        poly_verts = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = Polygon(domain).intersection(Polygon(poly_verts))  # trim cell by domain
        S[i] = i_cell.area
        C[i] = i_cell.centroid.coords[0]

        subsets = Delaunay(list(i_cell.exterior.coords))
        for j in range(len(subsets.simplices)):
            element = Polygon(subsets.points[index]
                              for index in subsets.simplices[j])
            s = element.area
            c = Point(element.centroid.coords[0])
            d = c.distance(Point(pts[i]))
            ret += s * d * d
        ret /= (r[i] + 1)
    return ret


def G(X):
    ret = np.zeros((camera_num, 2))
    pts = np.reshape(X, (camera_num, 2))
    pts = np.vstack((pts, domain))
    for i in range(camera_num):
        ret[i] = 2 * S[i] * (pts[i] - C[i]) / (r[i] + 1)
    return ret.flatten()


count = 0


def cbf(X):
    global count
    plt.figure(figsize=(6, 6))
    plt.cla()

    pts = np.reshape(X, (camera_num, 2))
    pts = np.vstack((pts, domain))
    voronoi_plot_2d(Voronoi(pts), ax=plt.gca(), show_vertices=False)

    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([0, 1])
    plt.gca().set_ylim([0, 1])
    plt.savefig('output/' + str(count).zfill(2) + '.png', bbox_inches='tight')
    count += 1


if __name__ == '__main__':
    random.seed(1224)
    X0 = [[random.random(), random.random()] for i in range(camera_num)]

    minimize(F, X0, method="L-BFGS-B", jac=G, callback=cbf,
             options={'maxiter': 100})
