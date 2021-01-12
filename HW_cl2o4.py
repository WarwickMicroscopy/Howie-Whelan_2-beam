import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

kernel_disp = '''
float3 mattranspose(__global float* mat) {

    float temp;
    temp = mat[1];
    mat[1] = mat[3];
    mat[3] = temp;
    temp = mat[2];
    mat[2] = mat[6];
    mat[6] = temp;
    temp = mat[5];
    mat[5] = mat[7];
    mat[7] = temp;
    //return mat;
}


float3 matmulvec(__global float* mat, float3 vec) {

    float3 out;
    out.x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
    out.y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
    out.z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
    return out;
}


__kernel void difference(__global float *in_out, __global float *in, float dz) {
    int xid = get_global_id(1);
    int zid = get_global_id(0);
    int xsiz = get_global_size(1);
    int id = xid + zid * xsiz;
    
    in_out[id] = (in[id] - in_out[id]) / dz;
}

__kernel void displacement(__global float *image_out,
                           __global float *c2d, __global float *d2c,
                           float pix2nm, float3 u, float3 g, float3 b, float3 b_unit, float3 b_edge_d,
                           float b_screw, float dt, float dz, float nu,
                           float phi, float psi, float theta) {

    // x-coordinate
    int xid = get_global_id(1);
    // z-coordinate
    int zid = get_global_id(0);
    // x-dimension
    int xsiz = get_global_size(1);
    // z-dimension
    int zsiz = get_global_size(0);
    // position of this pixel in the 1D array
    int id = xid + zid * xsiz;
    
    // vector from dislocation to this pixel, in nm
    float xR = ((float) xid + 0.5f - (float) xsiz / 2)*pix2nm;
    float til = sin(phi) + (float) xsiz * tan(psi)/( 2* (float) zsiz);
    float yR = (((float) zid + 0.5f + dz - (float) zsiz / 2)*til*dt )*pix2nm + xR*tan(theta);
    float3 r_d = {xR, yR, 0.0f};
    float r_mag = sqrt(dot(r_d, r_d));
    //cos(theta) & sin(theta) relative to Burgers vector
    float ct = dot(r_d, b_unit) / r_mag;
    float3 tu = (float3) {0.0f, 0.0f, 1.0f};
    float st = dot(tu, cross(b_unit, r_d) / r_mag);
    //Screw displacement
    float d_screw = b_screw * ( atan(r_d.y / (r_d.x)) - M_PI_F * (r_d.x < 0) ) / (2.0f * M_PI_F);
    float3 temp_1 = (float3) {0.0f, 0.0f, d_screw};
    float3 r_screw = matmulvec(d2c, temp_1);
    
    float3 temp_2 = b_edge_d * ct * st / (2.0f * M_PI_F * (1.0f - nu));
    float3 r_edge_0 = matmulvec(d2c, temp_2);
    
    float3 r_edge_1 = cross(b, u) * (((2.0f - 4.0f * nu) * log(r_mag) + (ct * ct - st * st)) / (8.0f * M_PI_F * (1.0f - nu)));
    
    float3 r_sum = r_screw + r_edge_0 + r_edge_1;
    
    //
    
    image_out[id] += dot(g, r_sum);
//    image_out[id] += zid;
}
'''


kernel_image = '''

float2 complexconj(float2 a) {
    return (float2) {a.x, -a.y};
}

float2 complexmul(float2 a, float2 b) {
    
    return (float2) { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
    
}

float4 matmulvec(float2* mat, float4 vec) {

    float4 out;

    out.xy = complexmul(mat[0], vec.xy) + complexmul(mat[1], vec.zw);
    out.zw = complexmul(mat[2], vec.xy) + complexmul(mat[3], vec.zw);
    
    return out;
    
}

__kernel void amplitude2intensity(__global float4 *in_out) {
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int xsiz = get_global_size(1);
    int id = xid * xsiz + yid;
    
    float2 f_0 = in_out[id].xy;
    float2 f_1 = in_out[id].zw;
    
    float2 bf = complexmul(f_0, complexconj(f_0));
    float2 df = complexmul(f_1, complexconj(f_1));
    
    in_out[id] = (float4) {bf, df};
}


__kernel void propagate_wave(__global float4 *in_out, __global float *sxz,
                              float3 g, float3 b, float3 nS,
                              int zsiz, float t, float dt, float pix2nm, 
                              int zlen, float s, float2 x_g, float x_0_i,
                              float phi, float psi, float theta, int k) {
                         
    float eps = 0.000000000001f;     
    int xid = get_global_id(0);
    int xsiz = get_global_size(0);
    int yid = get_global_id(1);
    int ysiz = get_global_size(1);
    // position of the current pixel in the linear array
    int id = xid * ysiz + yid;

    //top of the foil for this pixel
    float top = (zsiz-zlen)*( (float) yid / (float) ysiz);
    int h = (int) top;
    float m = top - h;
    
    //position of pixels in linear array
    int id_2 = (h + k) * xsiz + xid;
    int id_3 = (h + k + 1) * xsiz + xid;
    float s_local = s + (1.0f - m) * sxz[id_2] + m * sxz[id_3];
    
    float alpha = 0.0f;
    //if (xid > firs && xid < las && h + k - (int) (z_range / 2.0f) == 0.0f)
    //    alpha = 2.0f * M_PI_F * dot(g, b);
    
    // Howie-Whelan bit
    
    float x_g_r = x_g.x;
    float x_g_i = x_g.y;
    
    s_local += eps;
    float s_local2 = s_local * s_local;
    
    float recip_x_g_r = 1.0f / x_g_r;
    float recip_x_0_i_2 = 0.5f / x_0_i;
    
    float gamma_term = sqrt(s_local2 + recip_x_g_r * recip_x_g_r);
    float2 gamma = (float2) { (s_local - gamma_term) * 0.5f, (s_local + gamma_term) * 0.5f };
    
    float q_term = 0.5f / (x_g_i * sqrt(1.0f + (s_local * x_g_r) * (s_local * x_g_r)));
    float2 q = (float2) { recip_x_0_i_2 - q_term, recip_x_0_i_2 + q_term };
    
    float beta_2 = 0.5f * acos( s_local * x_g_r / sqrt( 1.0f + s_local2 * x_g_r * x_g_r ) );
    
    float2 sin_beta_2 = (float2) {sin(beta_2), 0.0f};
    float2 cos_beta_2 = (float2) {cos(beta_2), 0.0f};
    
    float2 exp_alpha = (float2) {cos(alpha), sin(alpha)};
    
    float2 big_c[4] = {cos_beta_2, sin_beta_2, -sin_beta_2.x * exp_alpha, cos_beta_2.x * exp_alpha};
    float2 big_c_t[4] = {big_c[0], big_c[2], big_c[1], big_c[3]};
    
    float2 big_g_g = 2.0f * M_PI_F * gamma * dt * pix2nm;
    float2 big_g_q = -2.0f * M_PI_F * q * dt * pix2nm;
    
    float2 big_g_0 = exp(big_g_q.x) * (float2) { cos(big_g_g.x), sin(big_g_g.x) };
    float2 big_g_3 = exp(big_g_q.y) * (float2) { cos(big_g_g.y), sin(big_g_g.y) };
    
    float2 big_g[4] = {big_g_0, 0.0f, 0.0f, big_g_3};
    
    float4 out = matmulvec(big_c, matmulvec(big_g, matmulvec(big_c_t, in_out[id])));
//    float4 out = {top,0.0f,yid, 0.0f};
    in_out[id] = out;
}
'''


class ClHowieWhelan:

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self.disp_r_prog = cl.Program(self.ctx, kernel_disp).build()
        self.image_prog = cl.Program(self.ctx, kernel_image).build()

        self.sxz_buf = None
        

    def calculate_deviations(self, xsiz, zsiz, pix2nm, dt, u, g, b, c2d, d2c, nu, phi, psi, theta):
        #fills up sxz array
        #set up buffers to pass between python and C++
        buf_size = xsiz * (zsiz+1) * 4

        mf = cl.mem_flags
        self.sxz_buf = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        out_buf_2 = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        c2d_buf = cl.Buffer(self.ctx, mf.READ_ONLY, c2d.size * 4)
        d2c_buf = cl.Buffer(self.ctx, mf.READ_ONLY, d2c.size * 4)

        shape = np.array([(zsiz+1), xsiz], dtype=np.int32)

        # set up contiguous buffers
        c2d_ = np.ascontiguousarray(c2d.ravel().astype(np.float32))
        cl.enqueue_copy(self.queue, c2d_buf, c2d_)
        d2c_ = np.ascontiguousarray(d2c.ravel().astype(np.float32))
        cl.enqueue_copy(self.queue, d2c_buf, d2c_)
        # fill with zeros as we add to initial buffer (makes handling the 2 bs easier)
        cl.enqueue_fill_buffer(self.queue, self.sxz_buf, np.float32(0.0), 0, buf_size)
        cl.enqueue_fill_buffer(self.queue, out_buf_2, np.float32(0.0), 0, buf_size)

        # the actual calculation
        #small change in z-coordinate to get derivative
        dz = 0.0
        #calculate x-z array of displacements
        self.displace_r(shape, self.sxz_buf, c2d_buf, d2c_buf, pix2nm, u, g, b, c2d, nu, phi, psi, theta, dt, dz)
        # calculate second array at a small z-shift
        dz = 0.01
        self.displace_r(shape, out_buf_2, c2d_buf, d2c_buf, pix2nm, u, g, b, c2d, nu, phi, psi, theta, dt, dz)

        # subtract one from the other to get the gradient
        dz_32 = np.float32(dz)
        self.disp_r_prog.difference(self.queue, shape, None, self.sxz_buf, out_buf_2, dz_32)
                

    def displace_r(self, shape, out_buf, c2d_buf, d2c_buf, pix2nm, u, g, b, c2d, nu, phi, psi, theta, dt, dz):
        #this simply converts the variables into float32 type and sends it off to the C++ subroutine
        b_screw = np.dot(b, u)
        b_edge = c2d @ (b - b_screw * u)  # NB a vector
        b_unit = b_edge / (np.dot(b_edge, b_edge) ** 0.5)

        # float3 is actually a float4 in disguise?
        nu_32 = np.float32(nu)
        dt_32 = np.float32(dt)
        dz_32 = np.float32(dz)
        b_screw_32 = np.float32(b_screw)
        pix2nm_32 = np.float32(pix2nm)
        phi_32 = np.float32(phi)
        psi_32 = np.float32(psi)
        theta_32 = np.float32(theta)
        u_32 = np.append(u, 0.0).astype(np.float32)
        g_32 = np.append(g, 0.0).astype(np.float32)
        b_32 = np.append(b, 0.0).astype(np.float32)
        b_unit_32 = np.append(b_unit, 0.0).astype(np.float32)
        b_edge_32 = np.append(b_edge, 0.0).astype(np.float32)

        self.disp_r_prog.displacement(self.queue, shape, None, out_buf,
                                      c2d_buf, d2c_buf, 
                                      pix2nm_32, u_32, g_32, b_32, b_unit_32, b_edge_32, b_screw_32,
                                      dt_32, dz_32, nu_32, phi_32, psi_32, theta_32)


    def get_sxz_buffer(self, xsiz, zsiz):
        output = np.zeros((zsiz, xsiz), dtype=np.float32)
        cl.enqueue_copy(self.queue, output, self.sxz_buf)

        return output


    def calculate_image(self, xsiz, ysiz, zsiz, pix2nm, t, dt, s,
                        Xg, X0i, g, b, nS, psi, theta, phi):
        if self.sxz_buf is None:
            raise Exception('sxz buffer has not been created. Did you run calculate_displacements?')

        mf = cl.mem_flags
        buf_size = ysiz * xsiz * 4 * 4
        in_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, buf_size)
        shape = np.array([xsiz, ysiz], dtype=np.int32)

        # Complex wave amplitudes are held in F = [BF,DF]
        # generate F values at the top surface [1,0]
        tableau = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        full = np.tile(tableau, xsiz*ysiz)
        cl.enqueue_copy(self.queue, in_out_buf, full.astype(np.float32))
        
        # Variables suitable for C++
        zsiz_i32 = np.int32(zsiz)
        t_32 = np.float32(t)
        dt_32 = np.float32(dt)
        pix2nm_32 = np.float32(pix2nm)
        s_32 = np.float32(s)
        x_g_32 = np.complex64(Xg)  # complex64 as it is 2 32-bit numbers
        x_0_i_32 = np.float32(X0i)
        g_32 = np.append(g, 0.0).astype(np.float32)
        b_32 = np.append(b, 0.0).astype(np.float32)
        nS_32 = np.append(nS, 0.0).astype(np.float32)
        phi_32 = np.float32(phi)
        psi_32 = np.float32(psi)
        theta_32 = np.float32(theta)


        #number of steps in wave propagation
        zlen_32 = np.int32(t*nS[2]/dt + 0.5)
        for k in range(zlen_32):
            k_i32 = np.int32(k)

            self.image_prog.propagate_wave(self.queue, shape, None, in_out_buf, self.sxz_buf,
                                        g_32, b_32, nS_32, zsiz_i32, t_32, dt_32, pix2nm_32, 
                                        zlen_32, s_32, x_g_32, x_0_i_32, phi_32, psi_32, theta_32, k_i32)

        self.image_prog.amplitude2intensity(self.queue, shape, None, in_out_buf)

        output = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, output, in_out_buf)

        image_bf = np.flip(output[:, :, 0], axis=0)
        image_df = np.flip(output[:, :, 2], axis=0)

        # image_bf = output[:, :, 0]
        # image_df = output[:, :, 2]

        return image_bf, image_df
