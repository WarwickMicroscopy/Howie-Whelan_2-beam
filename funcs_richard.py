import numpy as np

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

def displaceR(xyz,b,u,c2d,d2c, nu):
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

def calculate_displacements(b1, b2, xmax, zrange, dt, dz, s2c, c2d, d2c, p1, p2, u, nu, g):
    sxz = np.zeros((zrange, xmax), dtype='f')#32-bit for .tif saving

    # calculation of displacements
    # to avoid additional calculation: one dislocation
    if (abs(b2[0])+abs(b2[1])+abs(b2[2])<eps):
        for i in range (xmax):
            #looping over z with ymax steps
            for k in range(zrange):
                #coord of current voxel relative to centre in simulation frame is xyz
                v = np.array((i, 0, k*dt))
                #first dislocation
                xyz = s2c @ (v-p1)
                R = displaceR( xyz, b1, u, c2d, d2c, nu )
                gdotR = np.dot(g,R)
                #dislocation 1 at dz
                xyz = s2c @ (v-p1+dz)
                Rdz = displaceR( xyz, b1, u, c2d, d2c, nu )
                gdotRdz = np.dot(g,Rdz)
                sxz[k,i] = (gdotRdz - gdotR)/dz[2]
    else:#two dislocations
        for i in range (xmax):
            #looping over z with ymax steps
            for k in range(zrange):
                #coord of current voxel relative to centre in simulation frame is xyz
                v = np.array((i, 0, k*dt))
                #first dislocation
                xyz = s2c @ (v-p1)
                R1 = displaceR( xyz, b1, u, c2d, d2c, nu )
                #second dislocation
                xyz = s2c @ (v-p2)
                R2 = displaceR( xyz, b2, u, c2d, d2c, nu )
                R = R1 + R2
                gdotR = np.dot(g,R)
                #dislocation 1 at dz
                xyz = s2c @ (v-p1+dz)
                R1 = displaceR( xyz, b1, u, c2d, d2c, nu )
                #second dislocation
                xyz = s2c @ (v-p2+dz)
                R2 = displaceR( xyz, b2, u, c2d, d2c, nu )
                Rdz = R1 + R2
                gdotRdz = np.dot(g,Rdz)
                sxz[k,i] = (gdotRdz - gdotR)/(dz[2])

    return sxz

# def calculate_image(sxz, xmax, ymax, zmax, zrange, hpad, dt, phi, s, firs, las, Xg, X0i, g, b1):
#     Ib=np.zeros((xmax, ymax),dtype='f')#32-bit for .tif saving
#     # Dark field image
#     Id=np.zeros((xmax, ymax),dtype='f')
#     # Complex wave amplitudes are held in F = [BF,DF].
#     F0=np.array([[1], [0]])#input wave, top surface
#
#
#     for i in range (xmax):
#         for j in range (ymax):
#             F=F0[:]
#             #z loop integrates over a portion of sxz starting at h
#             hpos = (2*hpad+zmax-j/(dt*np.tan(phi)))#in pixels
#             h=int(hpos)
#             #linear interpolation between calculated points
#             m = hpos-h
#             for k in range(zmax):
#                 slocal = s + (1-m)*sxz[h+k,i]+m*sxz[h+k+1,i]
#                 #stacking fault shift is present between the two dislocations
#                 if (i>firs and i<las and (h+k-int(zrange/2)==0)):
#                     alpha = 2*np.pi*np.dot(g,b1)
#                 else:
#                     alpha = 0.0
#                 F=howieWhelan(F,Xg,X0i,slocal,alpha,dt)
#             # bright field is the first element times its complex conjugate
#             Ib[xmax-i-1,j]=(F[0]*np.conj(F[0])).real
#             # dark field is the second element times its complex conjugate
#             Id[xmax-i-1,j]=(F[1]*np.conj(F[1])).real
#
#     return Ib, Id

def calculate_image(sxz, xmax, ymax, zmax, zrange, hpad, dt, phi, s, firs, las, Xg, X0i, g, b1):
    Ib = np.zeros((xmax, ymax), dtype='f')  # 32-bit for .tif saving
    # Dark field image
    Id = np.zeros((xmax, ymax), dtype='f')
    # Complex wave amplitudes are held in F = [BF,DF].
    F0 = np.array([[1], [0]])  # input wave, top surface

    for i in range(xmax):
        for j in range(ymax):
            F = F0[:]
            # z loop integrates over a portion of sxz starting at h
            hpos = (2*hpad+zmax-j/(dt*np.tan(phi))) # in pixels
            h=int(hpos)
            # linear interpolation between calculated points
            m = hpos-h
            for k in range(zmax):
                slocal = s + (1-m)*sxz[h+k,i]+m*sxz[h+k+1,i]

                # stacking fault shift is present between the two dislocations
                if firs < i < las and h+k-int(zrange / 2) == 0:
                    alpha = 2*np.pi*np.dot(g,b1)
                else:
                    alpha = 0.0
                F = howieWhelan(F,Xg,X0i,slocal,alpha,dt)

            # bright field is the first element times its complex conjugate
            Ib[xmax-i-1,j] = (F[0]*np.conj(F[0])).real
            # dark field is the second element times its complex conjugate
            Id[xmax-i-1,j] = (F[1]*np.conj(F[1])).real

    return Ib, Id