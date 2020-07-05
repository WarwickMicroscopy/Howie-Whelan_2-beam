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

@author: Richard Beanland
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from libtiff import TIFF as tif

# to avoid division by zero errors
eps = 0.000000000001

def howieWhelan(F_in,Xg,X0i,s,alpha,t):
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
                [-np.sin(beta/2)*np.exp(complex(0,alpha)),
                 np.cos(beta/2)*np.exp(complex(0,alpha))]])
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
X0i = 100.0#nm
# Xg is the extinction distance for g (complex)
# The imaginary part should be larger than X0i
Xg = 20.0 + 1j*X0i*1.1#nm

# deviation parameter (typically between -0.1 and 0.1)
s = 0.0

# crystal thickness, nm
t = 150#nm

#integration step, nm
dt = 0.5#nm

# pixel scale
pix2nm = 1#nm per pixel

# default number of pixels arounnd the dislocation
pad = 40#pixels
# max image size in pixels & nm
# NB MUST be an even number since the dislocation lies at the boundary between pixels
# will OVER-RULE pad to make the image smaller
picmax = 600#pixels

# lattice parameter nm
a0 = 0.4

# Poisson's ratio
nu = 0.3

## Vector inputs
# NB cubic crystals only! Everything here in the crystal reference frame

# Burgers vectors: two partial dislocations
si=1/6
b1 = np.array((si,-si,-2*si))
b2 = np.array((2*si,si,-si))

# position of centre point of dislocation line relative to centre of volume
# ((1,0,0)) is vertical up, ((0,1,0)) is horizontal right, ((0,0,1)) is horiz. left
ti=int(pad/2)
q1 = np.array((-ti,0,0))#pixels
q2 = np.array((ti,0,0))

# line direction 
u = np.array((1,1,1))

# foil normal points along z parallel to the electron beam
z = np.array((1,1,0))

# g-vector 
g = np.array((-2,2,0))



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
    b1=-b1
    b2=-b2
# angle between dislocation and z-axis
phi = np.arccos(abs(np.dot(u,z)))
#check if they're parallel and use an alternative if so
if (abs(np.dot(u,z)-1) < eps):#they're parallel, set x parallel to b
    x = b1[:]
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

# g-vector on image is leng pixels long
leng=pad/4
gDisp = c2s @ g
gDisp = leng*gDisp/(np.dot(gDisp,gDisp)**0.5)
bDisp1 = c2s @ b1
bDisp1 = leng*bDisp1/(np.dot(bDisp1,bDisp1)**0.5)
bDisp2 = c2s @ b2
bDisp2 = leng*bDisp2/(np.dot(bDisp2,bDisp2)**0.5)

# g-vector magnitude, nm^-1
g = g/a0

#Burgers vector in nm
b1 = a0*b1
b2 = a0*b2

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
else:# dislocation is in the plane of the foil
    ymax = 2*pad
    hpad=0
zrange =2*(zmax+hpad) + 1#extra 1 for interpolation
# the height of the array for strain calcs
zdim = (zrange-1)*dt*pix2nm#in nm

##################################
# calculate strain field and hence
# x-z array of deviation parameters. The array has length zrange along z and
#we will integrate over a smaller length zmax. The integration range shifts 
# down for each y-pixel.
sxz=np.zeros((zrange,xmax),dtype='f')#32-bit for .tif saving
#small z value used to get derivative
dz = np.array((0,0,0.01))#pix
#point the dislocation passes through - the centre of simulation volume + q, in pixels
p1 = np.array((pad+0.5, 0.5, (hpad+zmax)*dt+0.5)) + q1
p2 = np.array((pad+0.5, 0.5, (hpad+zmax)*dt+0.5)) + q2
firs=min(p1[0],p2[0])
las=max(p1[0],p2[0])
# calculation of displacements
for i in range (xmax):
    #looping over z with ymax steps 
    for k in range(zrange):
        #coord of current voxel relative to centre in simulation frame is xyz
        v = np.array((i, 0, k*dt))
        #first dislocation
        xyz = s2c @ (pix2nm*(v-p1))
        R1 = displaceR( xyz, b1, u, c2d, d2c )
        #second dislocation
        xyz = s2c @ (pix2nm*(v-p2))
        R2 = displaceR( xyz, b2, u, c2d, d2c )
        R = R1 + R2
        gdotR = np.dot(g,R)
        #dislocation 1 at dz
        xyz = s2c @ (pix2nm*(v-p1+dz))
        R1 = displaceR( xyz, b1, u, c2d, d2c )
        #second dislocation
        xyz = s2c @ (pix2nm*(v-p2+dz))
        R2 = displaceR( xyz, b2, u, c2d, d2c )
        Rdz = R1 + R2
        gdotRdz = np.dot(g,Rdz)
        sxz[k,i] = (gdotRdz - gdotR)/(dz[2]*pix2nm)

##################################
# calculate image

# Previous versions had x horizontal y vertical (e.g. Ib[j,i])
# now switched (v1.3) to x vertical, increasing up; y horizontal increasing right
# z into the image (as seen on TEM screen).  Dislocation intersects top surface
# to the left, bottom surface to the right.
# Bright field image initialised as zeros
Ib=np.zeros((xmax, ymax),dtype='f')#32-bit for .tif saving
# Dark field image 
Id=np.zeros((xmax, ymax),dtype='f')
# Complex wave amplitudes are held in F = [BF,DF]. 
F0=np.array([[1], [0]])#input wave, top surface

# deviation parameter loop (typically between 0 and 0.1)
# for s_int in range (1):
#     s = 0.0+s_int*0.01
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
            #stacking fault shift is present between the two dislocations
            if (i>firs and i<las and (h+k-int(zrange/2)==0)):
                alpha = 2*np.pi*np.dot(g,b1)
            else:
                alpha = 0.0
            F=howieWhelan(F,Xg,X0i,slocal,alpha,dt)
        # bright field is the first element times its complex conjugate
        Ib[xmax-i-1,j]=(F[0]*np.conj(F[0])).real
        # dark field is the second element times its complex conjugate
        Id[xmax-i-1,j]=(F[1]*np.conj(F[1])).real

  
##################################    
    ### save & show the result ###
    
imgname="BF_t="+str(int(t))+"_s"+str(s)+".tif"
spoink = tif.open(imgname, mode='w')
spoink.write_image(Ib)
spoink.close
imgname="DF_t="+str(int(t))+"_s"+str(s)+".tif"
spoink = tif.open(imgname, mode='w')
spoink.write_image(Id)
spoink.close
imgname="dgdotRdz.tif"
spoink = tif.open(imgname, mode='w')
spoink.write_image(sxz)
spoink.close

fig=plt.figure(figsize=(8, 4))

fig.add_subplot(2, 1, 1)
plt.imshow(Ib)
plt.axis("off")
pt=int(pad/2)
plt.arrow(pt,pt,gDisp[1],-gDisp[0], shape='full', head_width=3, head_length=6)
plt.annotate("g", xy=(pt+2, pt+2))
fig.add_subplot(2, 1, 2)
plt.imshow(Id)
plt.axis("off")
if ((abs(bDisp1[0])+abs(bDisp1[1])) < eps):#Burgers vector is along z
    plt.annotate(".", xy=(pt,pt))
else:
    plt.arrow(pt,pt,bDisp1[1],-bDisp1[0], shape='full', head_width=3, head_length=6)    
    plt.arrow(pt,3*pt,bDisp2[1],-bDisp2[0], shape='full', head_width=3, head_length=6)    
plt.annotate("b1", xy=(pt+2, pt+2))
plt.annotate("b2", xy=(pt+2, 3*pt+2))
bbox_inches=0
plotnameP="t="+str(int(t))+"_s"+str(s)+".png"
print(plotnameP)
plt.savefig(plotnameP)#, format = "tif")

tic = time.perf_counter()
print("time",tic-toc)