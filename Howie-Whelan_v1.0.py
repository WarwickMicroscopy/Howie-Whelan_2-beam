# -*- coding: utf-8 -*-
"""
Created on Thursday Jun 11 2020

A bit of 2-beam diffraction contrast never hurt anyone
Based on Hirsch, Howie, Nicolson, Pashley and Whelan p207
and Head, Humble, Clarebrough, Morton, Forwood p31

Image calculations for a dislocation:
ignoring dilational components so that everything can be expressed as a local 
change in deviation parameter s

@author: Richard Beanland
"""


import numpy as np
import matplotlib.pyplot as plt
import time

# avoid division by zero error
eps = 0.000000000001

def howieWhelan(F_in,Xg,X0i,s,t):
    # All dimensions in nm
    Xgr=Xg.real
    Xgi=Xg.imag

    s=s+eps

    gamma=np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])
    q=np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)), 
                (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])
    beta=np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))
    #scattering matrix
    C=np.array([[np.cos(beta/2), np.sin(beta/2)],
                [-np.sin(beta/2), np.cos(beta/2)]])
    #inverse of C is just its transpose
    Ci=np.transpose(C)
    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0], 
                [0, np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]])
    F_out = C @ G @ Ci @ F_in
    #print(F_out)
    return F_out

def displaceR(xyz,b,u,c2d,d2c):
    # returns displacement vector R at coordinate xyz
    # all inputs are in the crystal reference frame
    bscrew = np.dot(b,u)#NB a scalar
    bedge = b - bscrew*u#NB a vector
    bedgeD = c2d @ bedge
    # vector to dislocation r is xyz minus the component parallel to u
    r = xyz-u*np.dot(xyz,u)
    r2 = np.dot(r,r)
    rmag = r2**0.5
    # vector to dislocation, in the dislocation frame
    rD = c2d @ r
    # From Head et al. Eq. 2.31, p31
    # infinite screw dislocation displacement is b.theta/(2pi):
    # using x=r.sin(theta) & y=r.cos(theta)
    # have to take away pi to avoid double-valued tan
    # phase wrapping bug for inclined dislocations is here
    Dscrew = bscrew*(np.arctan(rD[1]/rD[0])-np.pi*(rD[0]<0))/(2*np.pi)
    # screw displacement in the crystal frame
    Rscrew = d2c @ np.array((0,0,Dscrew))
    # infinite edge dislocation displacement field part 1: b.sin2theta/2(1-nu)
    Redge0 = d2c @ (bedgeD*rD[1]*rD[0]/(4*np.pi*r2*(1-nu)) )
    # part 2: (b x u)((1-2v)ln(r)/2(1-v)+cos2theta/4(1-v))
    Redge1 = np.cross(b,u)*( ( (2-4*nu)*np.log(rmag)+
                              (rD[1]**2-rD[0]*2)/r2 )/(4-4*nu))
    # total displacement in the crystal frame
    R = Rscrew + Redge0 + Redge1
    return R#[R,Dscrew]




toc = time.perf_counter()
### input variables ###

# Extinction distances
# X0i is the imaginary part of the 000 extinction distance
# thickness fringes disappear at about X0i nm
X0i = 400#nm
# Xg is the extinction distance for g (complex)
# The imaginary part should be larger than X0i
Xg = 20 + 1j*X0i*1.1#nm

# crystal thickness
# we assume voxel sizes to be 1x1x1 nm
t = 200#nm

# Calculate in INTEGER slices of thickness dt (nm)
# dt>1 speeds up calculation at the expense of poor results very close to the dislocation
dt = 1#nm

# deviation parameter (typically between 0 and 0.1)
s = 0.05

# default number of pixels arounnd the dislocation
pad = 30
# max image size in pixels & nm
# NB MUST be an even number since the dislocation lies at the boundary between pixels
# will OVER-RULE pad to make the image smaller
picmax = 200

# lattice parameter nm
a0 = 0.4

# Poisson's ratio
nu = 0.3

## Vector inputs

# NB cubic crystals only! Everything here in the crystal reference frame
# x-direction Miller indices
x=np.array((1,1,0))
# y-direction Miller indices
y=np.array((0,0,1))

# g-vector 
g = np.array((2,2,0))
# Burgers vector
b = np.array((0.5,0,0.5))

# line direction 
u = np.array((1,-1,1))



### setup calculations ###

# transformation matrices

# Change x & y to unit vectors and get z from the cross product
# NB there is no check to ensure x and y are perpendicular!!
x = x/(np.dot(x,x)**0.5)
y = y/(np.dot(y,y)**0.5)
z = np.cross(x,y)
#transformation matrix between simulation frame & crystal frame
c2s = np.array((x,y,z))
s2c = np.transpose(c2s)

#normalise line direction
u = u/(np.dot(u,u)**0.5)


# x-direction in the dislocation frame xD is parallel to u x z
# in the crystal ref frame this is
xDc = np.cross(u,s2c@z)
if (abs(np.dot(xDc,xDc)**0.5)<eps):#u and z are colinear, so just use simulation x
    xDc = x
else:#normalise it
    xDc = xDc/(np.dot(xDc,xDc)**0.5)
yDc = np.cross(u,xDc)
#transformation matrix between crystal frame & dislocation frame
c2d = np.array((xDc,yDc,u))
d2c = np.transpose(c2d)

# g-vector magnitude, nm^-1
g = g/a0

#Burgers vector magnitudes, nm
b = a0*b
bmag = np.dot(b,b)**0.5
bscrew = np.dot(b,u)#NB a scalar
bedge = b - bscrew*u#NB a vector
bedgeD = c2d @ bedge
bedgeX = bedgeD[0]
bedgeY = bedgeD[1]

# dislocation passes through the mid-point of the volume

# image dimensions are length of dislocation line projection in x and y
# plus pad pixels (nm) on each edge

if (abs(np.dot(u,z))>eps):#dislocation is not in the plane of the foil
    xmax = int(abs(t*np.dot(u,x)/np.dot(u,z)) + 2*pad)
    ymax = int(abs(t*np.dot(u,y)/np.dot(u,z)) + 2* pad)
else:
    xmax = picmax
    ymax = picmax
#crop it down to the specified maximum
if (xmax>picmax):
    xmax = picmax
if (ymax>picmax):
    ymax = picmax


# Bright field image initialised as zeros
Ib=np.zeros((ymax, xmax))
# Dark field image 
Id=np.zeros((ymax, xmax))

# other images, for debugging
P1=np.zeros((ymax, xmax))
P2=np.zeros((ymax, xmax))

# Complex wave amplitudes are held in F = [BF,DF]. 
F0=np.array(( (1), (0) ))#input wave, top surface
# initialise for calculation loop
F=np.array(( (1), (0) ))

# small change in z to obtain gradient
dz = dt#odd results if dz<>dt!


# calculation loop over thickness
for i in range (xmax):
    for j in range (ymax):
        F=F0
        for k in range(0,t,dt):
            #coord of current voxel relative to centre in simulation frame is uvw
            #NB extra 0.5 pixel is for even pixel size images, odd pixel case still to be done
            vX = i - xmax/2 + 0.5
            vY = j - ymax/2 + 0.5
            vZ = k - t/2 + 0.5
            R = displaceR( (s2c @ (vX,vY,vZ)), b, u, c2d, d2c )
            Rdz = displaceR( (s2c @ (vX,vY,vZ+dz)), b, u, c2d, d2c )          
            #vector differential of displacements from HHNPW pg 251
            dRdz = (R - Rdz)/dz
            slocal = s + np.dot(g,dRdz)
            #print("s",slocal)
            F=howieWhelan(F,Xg,X0i,slocal,dz)
        Fz=F    
        # dark field is the second element times its complex conjugate
        Id[j,i]=(Fz[1]*np.conj(Fz[1])).real
        # bright field is the first element times its complex conjugate
        Ib[j,i]=(Fz[0]*np.conj(Fz[0])).real

### show the result ###

fig=plt.figure(figsize=(8, 8))

fig.add_subplot(1, 2, 1)
plt.imshow(Ib)
plt.axis("off")
fig.add_subplot(1, 2, 2)
plt.imshow(Id)
plt.axis("off")
bbox_inches=0

tic = time.perf_counter()
print("time",tic-toc)