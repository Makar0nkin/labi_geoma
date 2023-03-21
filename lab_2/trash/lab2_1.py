import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import math


def drawLine(axes, x1, y1, x2, y2):
    line = Line2D([x1, x2], [y1, y2], color="k")
    axes.add_line(line)
    # plt.text(0.5, 1.1, "Line2D", horizontalalignment="center")


def drawPoint(x, y):
    plt.scatter(x, y)


def checkCords(pi, pi1,  p0):
    if (pi[0] <= p0[0] <= pi1[0] and pi[1] <= p0[1] <= pi1[1]) \
            or pi[0] <= p0[0] <= pi1[0] and pi1[1] <= p0[1] <= pi[1] \
            or pi1[0] <= p0[0] <= pi[0] and pi1[1] <= p0[1] <= pi[1] \
            or pi1[0] <= p0[0] <= pi[0] and pi[1] <= p0[1] <= pi1[1]:
        return False
    else:
        return -100


def getDet(x1, y1, x2, y2, x0, y0):
    a = np.array([[x2 - x1, y2 - y1], [x0 - x1, y0 - y1]])
    determinant = np.linalg.det(a)
    return determinant


def getAngle(p0, pi, pi1):
    num = (pi[0] - p0[0])*(pi1[0] - p0[0]) + (pi[1] - p0[1])*(pi1[1] - p0[1])
    den1 = math.sqrt(math.pow(pi[0] - p0[0], 2) + math.pow(pi[1] - p0[1], 2))
    den2 = math.sqrt(math.pow(pi1[0] - p0[0], 2) + math.pow(pi1[1] - p0[1], 2))
    den = den1 * den2
    if den == 0:
        return checkCords(pi, pi1, p0)
    cosA = num / den
    a = math.acos(cosA)
    x1 = p0[0]
    y1 = p0[1]
    x2 = pi[0]
    y2 = pi[1]
    x = pi1[0]
    y = pi1[1]
    determinant = getDet(x1, y1, x2, y2, x, y)
    if determinant < 0:
        return -a
    elif determinant == 0:
        return checkCords(pi, pi1, p0)
    else:
        return a


def getRandomInt(min, max):
    return random.randint(min, max)


def createPolygon(sides, axes, plt):
    point = [getRandomInt(-5, 5), getRandomInt(-5, 5)]
    # point = [-1, -3]
    drawPoint(point[0], point[1])
    points = [[-4, 0], [-3, 3], [0, 4], [2, 3.5], [4, 0], [2.5, -3], [0, -4]]
    i = 0
    j = 1
    while i < len(points) - 1:
        drawLine(axes, points[i][0], points[i][1], points[j][0], points[j][1])
        i = i + 1
        j = j + 1
    drawLine(axes, points[-1][0], points[-1][1], points[0][0], points[0][1])
    plt.show()

    #------- гибридный


    maxX = -5
    minX = 5
    maxY = -5
    minY = 5
    for obj in points:
        if obj[0] > maxX:
            maxX = obj[0]
        if obj[0] < minX:
            minX = obj[0]
        if obj[1] > maxY:
            maxY = obj[1]
        if obj[1] < minY:
            minY = obj[1]
    if point[0] > maxX or\
        point[0] < minX or\
        point[1] > maxY or\
        point[1] < minY:
        print("Outside by hybrid")
        return points


    #------ угловой тест


    i = 0
    s = 0
    while i < sides:
        if i + 1 == len(points):
            a = getAngle(point, points[i], points[0])
            if a == -100:
                print("Outside")
                return points
            if a == False:
                print("On line inside")
                return points
            s = s + a
            i = i + 1
            break
        a = getAngle(point, points[i], points[i + 1])
        if a == -100:
            print("Outside")
            return points
        if a == False:
            print("On line inside")
            return points
        s += a
        i += 1
    print(s)
    if 6 < s or s < -6:
        print("Inside")
    else:
        print("Outside")
    return points


def createLinePoints():
    res = [getRandomInt(-5, 5), getRandomInt(-5, 5)]
    return res


if __name__ == '__main__':
    tempX = 0
    tempY = 0
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    axes = plt.gca()
    axes.set_aspect("equal")
    createPolygon(7, axes, plt)
    plt.show()

