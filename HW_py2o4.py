# -*- coding: utf-8 -*-
"""
Serial python calculation if there's no GPU

v1.0 taken out of Howie-Whelan_v1 Aug 2020
v1.1 Modified along with Howie-Whelan_v2.3 Dec 2020
 

@author: Richard Beanland, Jon Peters

"""
import numpy as np
import matplotlib.pyplot as plt

eps = 0.000000000001

def howieWhelan(F_in,Xg,X0i,s,alpha,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real
    Xgi = Xg.imag

    s = s + eps

    gamma = np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])

    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])

    beta = np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))

    #scattering matrix
    C=np.array([[np.cos(beta/2), np.sin(beta/2)],
                [-np.sin(beta/2)*np.exp(complex(0,alpha)),
                 np.cos(beta/2)*np.exp(complex(0,alpha))]])

    #inverse of C is just its transpose
    Ci=np.transpose(C)

    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0],
                [0, np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]])

    F_out = C @ G @ Ci @ F_in

    return F_out

def gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, g):
    # returns displacement vector R at coordinate xyz
    r2 = np.dot(rD,rD)
    rmag = r2**0.5
    #cos(theta) & sin(theta) relative to Burgers vector
    ct = np.dot(rD,beUnit)/rmag
    sbt = np.cross(beUnit,rD)/rmag
    st = sbt[2]
    # From Head et al. Eq. 2.31, p31
    # infinite screw dislocation displacement is b.phi/(2pi):
    # using x=r.sin(phi) & y=r.cos(phi)
    # have to take away pi to avoid double-valued tan
    Rscrew = np.array((0,0,bscrew*(np.arctan(rD[1]/rD[0])-np.pi*(rD[0]<0))/(2*np.pi)))
    # infinite edge dislocation displacement field part 1: b.sin2theta/4pi(1-nu)
    # using sin(2theta)=2sin(theta)cos(theta)
    Redge0 = bedge*ct*st/(2*np.pi*(1-nu))
    # part 2: (b x u)*( (1-2v)ln(r)/2(1-v) + cos(2theta)/4(1-v) )/2pi
    # using cos(2theta)= cos^2(theta) - sin^2(theta)
    Redge1 = bxu*( ( (2-4*nu)*np.log(rmag)+(ct**2-st**2) )/(8*np.pi*(1-nu)))
    # total displacement in the crystal frame
    R = d2c @ (Rscrew + Redge0 + Redge1)
    # dot product with g-vector
    gR = np.dot(g,R)
    return gR

def calculate_deviations(xsiz, zsiz, pix2nm, t, dt, u, g, b, c2d, nu, phi, psi, theta):
    # calculates the local change in deviation parameter s as the z-gradient of the displacement field
    
    #dislocation components in the dislocation reference frame
    bscrew = np.dot(b,u)#NB a scalar
    bedge = c2d @ (b - bscrew*u)#NB a vector
    beUnit = bedge/(np.dot(bedge,bedge)**0.5)#a unit vector
    bxu = c2d @ np.cross(b,u)
    #tranformation matrix from dislocation to crystal frame
    d2c = np.transpose(c2d)
    
    #the x-z array array of deviation parameters
    sxz = np.zeros((xsiz, zsiz+1), dtype='f')#32-bit for .tif saving, +1 is for interpolation

    # small z value used to get derivative
    dz = 0.01
    deltaz = np.array((0, dz, 0))

    # calculation of displacements R and the gradient of g.R
    for x in range (xsiz):
        for z in range(zsiz+1):
            # working in the dislocation frame here
            # vector to dislocation is rD, NB half-pixels puts the dislocation between pixels
            rX = 0.5+x-xsiz/2
            rD = np.array((rX,
                           dt*( (0.5+z-zsiz/2)*(np.sin(phi) + xsiz*np.tan(psi)/(2*zsiz)) )
                               + rX*np.tan(theta),
                           0))*pix2nm#in nm
            #Displacement R is calculated in the crystal frame
            gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, g )
            rDdz = rD + np.array((0, dt*dz, 0))*pix2nm#in nm
            gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, g )
            sxz[x,z] = (gRdz - gR)/dz
#            sxz[x,z] = np.sqrt(np.dot(rD,rD))
                
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(sxz)
    plt.axis("off")    
    return sxz


def calculate_image(sxz, xsiz, ysiz, zsiz, pix2nm, t, dt, s,
                    Xg, X0i, g, b, nS, psi, theta, phi):
    Ib = np.zeros((xsiz, ysiz), dtype='f')  # 32-bit for .tif saving
    # Dark field image
    Id = np.zeros((xsiz, ysiz), dtype='f')
    # Complex wave amplitudes are held in F = [BF,DF]
    F0 = np.array([[1], [0]])  # input wave, top surface
    # centre point of simulation frame is p
    p = np.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
    # length of wave propagation
    zlen=int(t*nS[2]/dt + 0.5)#remember nS[2]=cos(tilt angle)

    for x in range(xsiz):
        for y in range(ysiz):
            F = F0[:]
            # z loop propagates the wave over a z-line of sxz of length zlen 
            # corresponding to the thickness of the foil (t/dt slices)
            # For this row the top of the foil is
#            top = (zsiz-zlen)*(1 - y/ysiz)
            top = (zsiz-zlen)*y/ysiz 
            # if (nS[1] > 0):
            #     top = (zsiz-zlen)*(1 - y/ysiz + x*xsiz*np.tan(theta)) # in pixels
            # else:
            #     top = (zsiz-zlen)*(y/ysiz + x*xsiz*np.tan(theta))
            h=int(top)
            # linear interpolation of slocal between calculated points
            m = top-h
            for z in range(zlen):
                slocal = s + (1-m)*sxz[x,h+z]+m*sxz[x,h+z+1]

                # stacking fault shift is present between the two dislocations
                # if firs < x < las and h+z-int(zrange / 2) == 0:
                #     alpha = 2*np.pi*np.dot(g,b1)
                # else:
                alpha = 0.0
                F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm)

            # bright field is the first element times its complex conjugate
            Ib[xsiz-x-1,y] = (F[0]*np.conj(F[0])).real
            # dark field is the second element times its complex conjugate
            Id[xsiz-x-1,y] = (F[1]*np.conj(F[1])).real

    return Ib, Id