# -*- coding: utf-8 -*-
"""
Created on Thursday Jun 19 2020

A bit of 2-beam diffraction contrast never hurt anyone
Based on Hirsch, Howie, Nicolson, Pashley and Whelan p207
and Head, Humble, Clarebrough, Morton, Forwood p31

Image calculations for a dislocation:
ignoring dilational components so that everything can be expressed as a local 
change in deviation parameter s

Deviation parameter loop version

@author: Richard Beanland
"""


import numpy as np
import matplotlib.pyplot as plt
import time
from libtiff import TIFF as tif

# to avoid division by zero errors
eps = 0.000000000001

def howieWhelan(F_in,Xg,X0i,s,t):
    #for integration over n slices
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
    # all inputs are in nm, in the crystal reference frame
    bscrew = np.dot(b,u)#NB a scalar
    bedge = b - bscrew*u#NB a vector
    bedgeD = c2d @ bedge
    # vector to dislocation r is xyz minus the component parallel to u
    r = xyz-u*np.dot(xyz,u)
    r2 = np.dot(r,r)
    rmag = r2**0.5 + eps
    #cos(theta) & sin(theta) relative to Burgers vector
    beUnit = bedge/(np.dot(bedge,bedge)**0.5)
    ct = np.dot(r,beUnit)/rmag
    st = np.dot(u,np.cross(beUnit,r)/rmag)
    # vector to dislocation, in the dislocation frame
    rD = c2d @ r
    # From Head et al. Eq. 2.31, p31
    # infinite screw dislocation displacement is b.phi/(2pi):
    # using x=r.sin(phi) & y=r.cos(phi)
    # have to take away pi to avoid double-valued tan
    Dscrew = bscrew*(np.arctan(rD[1]/(rD[0]+eps))-np.pi*(rD[0]<0))/(2*np.pi)
    # screw displacement in the crystal frame
    Rscrew = d2c @ np.array((0,0,Dscrew))
    # infinite edge dislocation displacement field part 1: b.sin2theta/4pi(1-nu)
    Redge0 = d2c @ (bedgeD*ct*st/(4*np.pi*(1-nu)) )
    # part 2: (b x u)*( (1-2v)ln(r)/2(1-v) + cos(2theta)/4(1-v) )/2pi
    # using cos(2theta)= cos^2(theta) - sin^2(theta)
    Redge1 = np.cross(b,u)*( ( (2-4*nu)*np.log(rmag)+(ct**2-st**2) )/(8*np.pi*(1-nu)))
    # total displacement in the crystal frame
    R = Rscrew + Redge0 + Redge1
    return R




toc = time.perf_counter()
### input variables ###

# Extinction distances
# X0i is the imaginary part of the 000 extinction distance
# thickness fringes disappear at about X0i nm
X0i = 400.0#nm
# Xg is the extinction distance for g (complex)
# The imaginary part should be larger than X0i
Xg = 20.0 + 1j*X0i*1.1#nm

# deviation parameter (typically between -0.1 and 0.1)
s = 0.#now replaced with loop

# crystal thickness
# we assume voxel sizes to be 1x1x1 nm
t = 150#nm

#integration step
dt = 0.05

# pixel scale
pix2nm = 1#nm per pixel

# default number of pixels arounnd the dislocation
pad = 25
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

# Burgers vector
b = np.array((0.5,0.,0.5))

# line direction 
u = np.array((1,1,1))

# foil normal points along z parallel to the electron beam
z = np.array((0,0,1))

# g-vector 
g = np.array((2,-2,0))



### setup calculations ###

# x, y and z are the defining unit vectors of the simulation volume
# written in the crystal frame
# x is defined by the cross product of u and n
#normalise line direction
u = u/(np.dot(u,u)**0.5)
#normalise foil normal
z = z/(np.dot(z,z)**0.5)
# we want u pointing to the same side of the foil as z
if(np.dot(u,z)<0):#they're antiparallel, reverse u and b 
    u=-u
    b=-b
# angle between dislocation and z-axis
phi = np.arccos(abs(np.dot(u,z)))
#check if they're parallel and use an alternative if so
if (abs(np.dot(u,z)-1) < eps):#they're parallel, set x parallel to b
    x = b[:]
    x = x/(np.dot(x,x)**0.5)
    if (abs(np.dot(x,z)-1) < eps):#they're parallel too, set x parallel to g
        x = g[:]#this will fail for u=z=b=g but it would be stupid
else:
    x = np.cross(u,z)
x = x/(np.dot(x,x)**0.5)
# y is the cross product of z & x
y = np.cross(z,x)

#transformation matrices between simulation frame & crystal frame
c2s = np.array((x,y,z))
s2c = np.transpose(c2s)


# dislocation frame has zD parallel to u & xD parallel to x
# yD is given by their cross product
xD = x
yD = np.cross(u,x)
zD = u
#transformation matrix between crystal frame & dislocation frame
c2d = np.array((xD,yD,zD))
d2c = np.transpose(c2d)

# g-vector magnitude, nm^-1
g = g/a0

#Burgers vector in nm
b = a0*b

##################################
# image dimensions are length of dislocation line projection in y
# plus pad pixels on each edge
xmax = 2*pad# in pixels
xdim = xmax*pix2nm# in nm
# there are zmax steps over the thickness
zmax = int(0.5+t/dt)

if (abs(np.dot(u,z))>eps):#dislocation is not in the plane of the foil
    # y-length needed to capture the full length of the dislocation plus padding
    ydim = t*np.tan(phi) + 2*pix2nm*pad#in nm
    ymax = int(ydim/pix2nm)# in pixels
    if (ymax>picmax):
        ymax = picmax
    # corresponding thickness range
    # height padding
    hpad = int(0.5+pad/(dt*np.tan(phi)))#in pixels
    zrange =2*(zmax+hpad) + 1#extra 1 for interpolation
else:# dislocation is in the plane of the foil
    ymax = 2*pad
    hpad=0
    zrange = 2*zmax 
# the height of the array for strain calcs
zdim = (zrange-1)*dt*pix2nm#in nm

##################################
# calculate strain field and hence
# x-z array of deviation parameters
sxz=np.zeros((zrange,xmax))
#small z value used to get derivative
dz = 0.05#pix
# calculation of displacements
for i in range (xmax):
    #looping over z with ymax steps 
    for k in range(zrange):
        #coord of current voxel relative to centre in simulation frame is xyz
        xyzS = pix2nm*np.array((i-pad+0.5,0.5,0.5+k-(hpad+zmax)))#in nm
        xyz = s2c @ xyzS
        R = displaceR( xyz, b, u, c2d, d2c )
        xyzSdz = pix2nm*np.array((i-pad+0.5,0.5,0.5+k-(hpad+zmax)+dz))
        xyzdz = s2c @ xyzSdz
        Rdz = displaceR( xyzdz, b, u, c2d, d2c )
        dRdz = (R - Rdz)/dz
        sxz[k,i] = np.dot(g,dRdz)

##################################
# calculate image

# Bright field image initialised as zeros
Ib=np.zeros((ymax, xmax),dtype='f')#32-bit for .tif svaing
# Dark field image 
Id=np.zeros((ymax, xmax),dtype='f')
# Complex wave amplitudes are held in F = [BF,DF]. 
F0=np.array([[1], [0]])#input wave, top surface

# deviation parameter loop (typically between 0 and 0.1)
for s_int in range (16):
    s = -0.05+s_int*0.01
    # calculation loop over coords
    for i in range (xmax):
        for j in range (ymax):
            F=F0[:]
            #z loop integrates over a portion of sxz starting at h
            hpos = (2*hpad+zmax-j/(dt*np.tan(phi)))#in pixels
            h=int(hpos)
            #linear interpolation between calculated points
            m = hpos-h
            for k in range(zmax):
                slocal = s + (1-m)*sxz[h+k,i]+m*sxz[h+k+1,i]
                F=howieWhelan(F,Xg,X0i,slocal,dt)
            # dark field is the second element times its complex conjugate
            Id[j,i]=(F[1]*np.conj(F[1])).real
            # bright field is the first element times its complex conjugate
            Ib[j,i]=(F[0]*np.conj(F[0])).real

  
##################################    
    ### save & show the result ###
    
    imgname="BF_s="+str(s_int)+".tif"
    spoink = tif.open(imgname, mode='w')
    spoink.write_image(Ib)
    spoink.close
    imgname="DF_s="+str(s_int)+".tif"
    spoink = tif.open(imgname, mode='w')
    spoink.write_image(Id)
    spoink.close
    
    fig=plt.figure(figsize=(8, 4))
    
    fig.add_subplot(1, 2, 1)
    plt.imshow(Ib)
    plt.axis("off")
    fig.add_subplot(1, 2, 2)
    plt.imshow(Id)
    plt.axis("off")
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(sxz)
    bbox_inches=0
    plotnameP="s="+str(s_int)+".tif"
    print(plotnameP)
    plt.savefig(plotnameP, format = "tif")

tic = time.perf_counter()
print("time",tic-toc)