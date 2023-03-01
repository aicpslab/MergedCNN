import numpy as np
from star import Star
from box import Box
from matplotlib import pyplot as plt


def plot_2d_box(sets, x_pos, y_pos):

    xmin = np.zeros((1, 1))
    xmax = np.zeros((1, 1))
    ymin = np.zeros((1, 1))
    ymax = np.zeros((1, 1))

    if isinstance(sets, Star):
        xmin[0, 0], xmax[0, 0] = sets.getRange(x_pos)
        ymin[0, 0], ymax[0, 0] = sets.getRange(y_pos)

    lb = np.concatenate((xmin, ymin))#np.array([[xmin], [ymin]])
    ub = np.concatenate((xmax, ymax))#np.array([[xmax], [ymax]])
    boxes = Box(lb, ub)
    plotBoxes_2D(boxes, 0, 1)


def plotBoxes_2D(boxes, x_pos, y_pos):

    if isinstance(boxes, Box) is False:
        raise Exception("plot object is not a box")

    if (x_pos > boxes.dim) | (y_pos > boxes.dim):
        raise Exception("Invalid x_pos or y_pos")

    x = np.array([[boxes.lb[x_pos, 0], boxes.ub[x_pos, 0]], [boxes.lb[x_pos, 0], boxes.ub[x_pos, 0]]])
    y = np.array([[boxes.lb[y_pos, 0], boxes.lb[y_pos, 0]], [boxes.ub[y_pos, 0], boxes.ub[y_pos, 0]]])

    Z = np.eye(1)
    #np.ones(((boxes.ub[x_pos]-boxes.lb[x_pos]), (boxes.ub[y_pos]-boxes.lb[y_pos])))
    plt.pcolormesh(x, y, Z)
    plt.show()
