import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize

CAMERA_NUM = 5
DOMAIN = [[0, 0], [1, 0], [1, 1], [0, 1]]
EXTRA = [[-1, -1], [2, -1], [2, 2], [-1, 2]]

r = np.ones((CAMERA_NUM, 1))
for i in range(CAMERA_NUM):
    r[i] = (1 + i % 3) * 0.1

S = np.zeros((CAMERA_NUM, 1))
C = np.zeros((CAMERA_NUM, 2))

def F(X):
    global DOMAIN
    pts = np.reshape(X, (CAMERA_NUM, 2))
    pts = np.vstack((pts, EXTRA))
    vor = Voronoi(pts)
    ret = 0.0
    for i in range(CAMERA_NUM):
        poly_verts = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = Polygon(DOMAIN).intersection(Polygon(poly_verts))  # trim cell by DOMAIN
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
    ret = np.zeros((CAMERA_NUM, 2))
    pts = np.reshape(X, (CAMERA_NUM, 2))
    pts = np.vstack((pts, EXTRA))
    for i in range(CAMERA_NUM):
        ret[i] = 2 * S[i] * (pts[i] - C[i]) / (r[i] + 1)
    return ret.flatten()


count = 0
def cbf(X):
    global count
    plt.figure(figsize=(6, 6))
    plt.cla()

    pts = np.reshape(X, (CAMERA_NUM, 2))
    pts = np.vstack((pts, EXTRA))
    ax = plt.gcf().gca()
    voronoi_plot_2d(Voronoi(pts), ax=ax, show_vertices=False)

    for i in range(CAMERA_NUM):
        circle = plt.Circle(pts[i], r[i], alpha=0.1, color='blue')
        ax.add_artist(circle)

    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([0, 1])
    plt.gca().set_ylim([0, 1])
    # plt.savefig('output/' + str(count).zfill(2) + '.png', bbox_inches='tight')
    plt.show()
    count += 1

def plot_initial_state(X0):
    pts = np.vstack((X0, EXTRA))
    ax = plt.gcf().gca()
    voronoi_plot_2d(Voronoi(pts), ax=ax, show_vertices=False)

    for i in range(CAMERA_NUM):
        circle = plt.Circle(pts[i], r[i], alpha=0.1, color='blue')
        ax.add_artist(circle)

    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([0, 1])
    plt.gca().set_ylim([0, 1])
    plt.show()

if __name__ == '__main__':
    random.seed(1224)
    X0 = [[0.1 * random.random(), 0.1 * random.random()] for i in range(CAMERA_NUM)]

    plot_initial_state(X0)

    minimize(F, X0, method="L-BFGS-B", jac=G, callback=cbf,
             options={'maxiter': 100})
