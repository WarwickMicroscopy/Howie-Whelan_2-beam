# -*- coding: utf-8 -*-
"""
Created on Thursday Jun 19 2020

A bit of 2-beam diffraction contrast never hurt anyone
Based on Hirsch, Howie, Nicolson, Pashley and Whelan p207
and Head, Humble, Clarebrough, Morton, Forwood p31

Image calculations for a dislocation:
ignoring dilational components so that everything can be expressed as a local 
change in deviation parameter s

Dissociated dislocation version

OpenCL speed up by Jon Peters, about 50,000x faster on a Dell Inspiron 7577 laptop

@author: Richard Beanland, Jon Peters
"""

import numpy as np
import matplotlib.pyplot as plt
import time
# from libtiff import TIFF as tif
from PIL import Image

import funcs_richard as funcs_1
import funcs_opencl as funcs_3

# to avoid division by zero errors
eps = 0.000000000001


use_cl=True
save_images=True

if use_cl:
    suffix = "_cl"
else:
    suffix = ""

toc = time.perf_counter()
# input variables

# Extinction distances
# X0i is the imaginary part of the 000 extinction distance
# thickness fringes disappear at about X0i nm
X0i = 100.0  # nm
# Xg is the extinction distance for g (complex)
# The imaginary part should be larger than X0i
Xg = 20.0 + 1j * X0i * 1.1  # nm

# deviation parameter (typically between -0.1 and 0.1)
s = 0.0

# crystal thickness, nm
t = 50  # nm

# integration step, nm
dt = 0.5  # nm

# pixel scale
# want more or less pixels? this will change the image size
# with an according increase (<1) or decrease (>1) in calculation time
pix2nm = 0.1# nm per pixel

# default number of pixels arounnd the dislocation
pad = 40  # pixels

# max image size in pixels & nm
# NB MUST be an even number since the dislocation lies at the boundary between pixels
# will OVER-RULE pad to make the image smaller
picmax = 6000  # pixels

# lattice parameter nm
a0 = 0.4

# Poisson's ratio
nu = 0.3

# Half the distance between dislocations, if a pair
sep = 0#int(pad / 2)  # in pixels

# Vector inputs
# NB cubic crystals only! Everything here in the crystal reference frame

# Burgers vectors: two partial dislocations
si = 1 / 2
b1 = np.array((si, 0, si))
# b2 = np.array((0.,0.,0.))
b2 = np.array((0, 0, 0))

# line direction
u = np.array((1, 0, -1))

# foil normal points along z parallel to the electron beam
z = np.array((1, 1, 0))

# g-vector
g = np.array((1,-1,0))



# setup calculations

# scale dimensions
X0i = X0i / pix2nm
Xg = Xg / pix2nm
t = t / pix2nm
pad = int(pad / pix2nm + 0.5)
picmax = int(picmax / pix2nm + 0.5)
sep = int(sep / pix2nm + 0.5)
a0 = a0 / pix2nm

# position of centre point of dislocation line relative to centre of volume
# ((1,0,0)) is vertical up, ((0,1,0)) is horizontal right, ((0,0,1)) is horiz. left
q1 = np.array((-sep, 0, 0))  # pixels
q2 = np.array((sep, 0, 0))

# x, y and z are the defining unit vectors of the simulation volume
# written in the crystal frame
# x is defined by the cross product of u and n
# normalise line direction
u = u / (np.dot(u, u) ** 0.5)
# normalise foil normal
z = z / (np.dot(z, z) ** 0.5)
# we want u pointing to the same side of the foil as z
if np.dot(u, z) < 0:  # they're antiparallel, reverse u and b
    u = -u
    b1 = -b1
    b2 = -b2
# angle between dislocation and z-axis
phi = np.arccos(abs(np.dot(u, z)))
# check if they're parallel and use an alternative if so
if abs(np.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    x = b1[:]
    x = x / (np.dot(x, x) ** 0.5)
    if abs(np.dot(x, z) - 1) < eps:  # they're parallel too, set x parallel to g
        x = g[:]  # this will fail for u=z=b=g but it would be stupid
else:
    x = np.cross(u, z)
x = x / (np.dot(x, x) ** 0.5)
# y is the cross product of z & x
y = np.cross(z, x)

# transformation matrices between simulation frame & crystal frame
c2s = np.array((x, y, z))
s2c = np.transpose(c2s)

# dislocation frame has zD parallel to u & xD parallel to x
# yD is given by their cross product
xD = x
yD = np.cross(u, x)
zD = u
# transformation matrix between crystal frame & dislocation frame
c2d = np.array((xD, yD, zD))
d2c = np.transpose(c2d)

# g-vector on image is leng pixels long
leng = pad / 4
gDisp = c2s @ g
gDisp = leng * gDisp / (np.dot(gDisp, gDisp) ** 0.5)
bDisp1 = c2s @ b1
bDisp1 = leng * bDisp1 / (np.dot(bDisp1, bDisp1) ** 0.5)
if abs(b2[0]) + abs(b2[1]) + abs(b2[2]) > eps:
    bDisp2 = c2s @ b2
    bDisp2 = leng * bDisp2 / (np.dot(bDisp2, bDisp2) ** 0.5)

# g-vector magnitude, nm^-1
g = g / a0

# Burgers vector in nm
b1 = a0 * b1
b2 = a0 * b2

##################################
# image dimensions are length of dislocation line projection in y
# plus pad pixels on each edge
xmax = 2 * pad  # in pixels
# there are zmax steps over the thickness
zmax = int(0.5 + t / dt)

if abs(np.dot(u, z)) > eps:  # dislocation is not in the plane of the foil
    # y-length needed to capture the full length of the dislocation plus padding
    ymax = int(t * np.tan(phi) + 2 * pad)  # in pixels
    if ymax > picmax:
        ymax = picmax
    # corresponding thickness range
    # height padding
    hpad = int(0.5 + pad / (dt * np.tan(phi)))  # in pixels
else:  # dislocation is in the plane of the foil
    ymax = 2 * pad
    hpad = 0
zrange = 2 * (zmax + hpad) + 1  # extra 1 for interpolation
# the height of the array for strain calcs
zdim = (zrange - 1) * dt

##################################
# calculate strain field and hence
# x-z array of deviation parameters. The array has length zrange along z and
# we will integrate over a smaller length zmax. The integration range shifts
# down for each y-pixel.

# small z value used to get derivative
dz = np.array((0, 0, 0.01))  # pix
# point the dislocation passes through - the centre of simulation volume + q, in pixels
p1 = np.array((pad + 0.5, 0.5, (hpad + zmax) * dt + 0.5)) + q1
p2 = np.array((pad + 0.5, 0.5, (hpad + zmax) * dt + 0.5)) + q2

firs = min(p1[0], p2[0])
las = max(p1[0], p2[0])

start_time = time.perf_counter()

if use_cl:
    cl_hw = funcs_3.ClHowieWhelan()
    cl_hw.calculate_displacements(b1, b2, xmax, zrange, dt, dz, s2c, c2d, d2c, p1, p2, u, nu, g)
    Ib, Id = cl_hw.calculate_image(xmax, ymax, zmax, zrange, hpad, dt, phi, s, firs, las, Xg, X0i, g, b1)
else:
    sxz = funcs_1.calculate_displacements(b1, b2, xmax, zrange, dt, dz, s2c, c2d, d2c, p1, p2, u, nu, g)
    Ib, Id = funcs_1.calculate_image(sxz, xmax, ymax, zmax, zrange, hpad, dt, phi, s, firs, las, Xg, X0i, g, b1)

end_time = time.perf_counter()

duration = end_time - start_time

print("Main loops took: " + str(duration) + " seconds")

if save_images:

    # save & show the result
    t = t * pix2nm
    heady=6/pix2nm
    imgname = "BF_t=" + str(int(t)) + "_s" + str(s) + suffix + ".tif"
    Image.fromarray(Ib).save(imgname)
    imgname = "DF_t=" + str(int(t)) + "_s" + str(s) + suffix + ".tif"
    Image.fromarray(Id).save(imgname)

    fig = plt.figure(figsize=(8, 4))

    fig.add_subplot(2, 1, 1)
    plt.imshow(Ib)
    plt.axis("off")
    pt = int(pad / 2)
    plt.arrow(pt, pt, gDisp[1], -gDisp[0],
              shape='full', head_width=heady, head_length=2*heady)
    plt.annotate("g", xy=(pt + 2, pt + 2))
    fig.add_subplot(2, 1, 2)
    plt.imshow(Id)
    plt.axis("off")
    if (abs(bDisp1[0]) + abs(bDisp1[1])) < eps:  # Burgers vector is along z
        plt.annotate(".", xy=(pt, pt))
    else:
        plt.arrow(pt, pt, bDisp1[1], -bDisp1[0],
                  shape='full', head_width=heady, head_length=2*heady)
        if abs(b2[0]) + abs(b2[1]) + abs(b2[2]) > eps:
            plt.arrow(pt, 3 * pt, bDisp2[1], -bDisp2[0],
                      shape='full', head_width=heady, head_length=2*heady)
            plt.annotate("b2", xy=(pt + 2, 3 * pt + 2))
    plt.annotate("b1", xy=(pt + 2, pt + 2))
    bbox_inches = 0
    plotnameP = "t=" + str(int(t)) + "_s" + str(s) + suffix + ".png"
    # print(plotnameP)
    plt.savefig(plotnameP)  # , format = "tif")

tic = time.perf_counter()
print("Full function took: " + str(tic - toc) + " seconds")
