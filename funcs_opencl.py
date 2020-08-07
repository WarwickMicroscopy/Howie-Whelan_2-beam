import pyopencl as cl
import numpy as np


kernel_disp = '''
float3 matmulvec(__global float* mat, float3 vec) {

    float3 out;

    out.x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
    out.y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
    out.z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
    
    return out;
    
}

float vecdot(float3 v1, float3 v2) {

    //return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    return dot(v1, v2);
    
}

__kernel void difference(__global float *in_out, __global float *in, float dz) {
    int xid = get_global_id(1);
    int zid = get_global_id(0);
    int width = get_global_size(1);
    int id = xid + zid * width;
    
    in_out[id] = (in[id] - in_out[id]) / dz;
}

__kernel void displacement(__global float *image_out,
                           __global float *s2c, __global float *c2d, __global float *d2c,
                           float3 p1, float3 u, float3 g, float3 b, float3 b_unit, float3 b_edge_d,
                           float b_screw, float dt, float nu) {

    float eps = 0.000000000001f;
    int xid = get_global_id(1);
    int zid = get_global_id(0);
    
    int width = get_global_size(1);
    // int height = get_global_size(0);
    
    int id = xid + zid * width;
    
    float3 v = {(float) xid, 0.0f, (float) zid * dt};

    float3 xyz = matmulvec(s2c, v - p1);

    // displaceR
    
    float3 r = xyz - u * dot(xyz, u);
    float r_mag = sqrt(dot(r, r));
    
    float ct = dot(r, b_unit) / r_mag;
    float st = dot(u, cross(b_unit, r) / r_mag);
    
    float3 r_d = matmulvec(c2d, r);
    
    float d_screw = b_screw * ( atan(r_d.y / (r_d.x + eps)) - M_PI_F * (r_d.x < 0) ) / (2.0f * M_PI_F);
    
    float3 temp_1 = (float3) {0.0f, 0.0f, d_screw};
    float3 r_screw = matmulvec(d2c, temp_1);
    
    float3 temp_2 = b_edge_d * ct * st / (4.0f * M_PI_F * (1.0f - nu));
    float3 r_edge_0 = matmulvec(d2c, temp_2);
    
    float3 r_edge_1 = cross(b, u) * (((2.0f - 4.0f * nu) * log(r_mag) + (ct * ct - st * st)) / (8.0f * M_PI_F * (1.0f - nu)));
    
    float3 r_sum = r_screw + r_edge_0 + r_edge_1;
    
    //
    
    image_out[id] += dot(g, r_sum);
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

__kernel void calculate_image(__global float4 *in_out) {
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    int width = get_global_size(1);
    int id = xid * width + yid;
    
    float2 f_0 = in_out[id].xy;
    float2 f_1 = in_out[id].zw;
    
    float2 bf = complexmul(f_0, complexconj(f_0));
    float2 df = complexmul(f_1, complexconj(f_1));
    
    in_out[id] = (float4) {bf, df};
}

__kernel void calculate_f(__global float4 *in_out, __global float *sxz,
                              float3 g, float3 b,
                              float2 x_g,
                              float x_0_i, float z_range, float dt, float phi, float firs, float las, float s,
                              int k, int h_pad, int z_max) {
                         
    float eps = 0.000000000001f;     
    int xid = get_global_id(0);
    int yid = get_global_id(1);
    
    int width = get_global_size(1);
    int height = get_global_size(0);
    int id = xid * width + yid;
    
    float h_pos = 2.0f * h_pad + z_max - yid / (dt * tan(phi));
    int h = (int) h_pos;
    float m = h_pos - h;
    
    int id_2 = (h + k) * height + xid;
    int id_3 = (h + k + 1) * height + xid;
    float s_local = s + (1.0f - m) * sxz[id_2] + m * sxz[id_3];
    
    float alpha = 0.0f;
    if (xid > firs && xid < las && h + k - (int) (z_range / 2.0f) == 0.0f)
        alpha = 2.0f * M_PI_F * dot(g, b);
    
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
    
    float2 big_g_g = 2.0f * M_PI_F * gamma * dt;
    float2 big_g_q = -2.0f * M_PI_F * q * dt;
    
    float2 big_g_0 = exp(big_g_q.x) * (float2) { cos(big_g_g.x), sin(big_g_g.x) };
    float2 big_g_3 = exp(big_g_q.y) * (float2) { cos(big_g_g.y), sin(big_g_g.y) };
    
    float2 big_g[4] = {big_g_0, 0.0f, 0.0f, big_g_3};
    
    float4 out = matmulvec(big_c, matmulvec(big_g, matmulvec(big_c_t, in_out[id])));
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

    def calculate_displacements(self, b1, b2, xmax, zrange, dt, dz, s2c, c2d, d2c, p1, p2, u, nu, g):

        mf = cl.mem_flags
        self.sxz_buf = cl.Buffer(self.ctx, mf.READ_WRITE, zrange * xmax * 4)
        out_buf_2 = cl.Buffer(self.ctx, mf.READ_WRITE, zrange * xmax * 4)
        s2c_buf = cl.Buffer(self.ctx, mf.READ_ONLY, s2c.size * 4)
        c2d_buf = cl.Buffer(self.ctx, mf.READ_ONLY, c2d.size * 4)
        d2c_buf = cl.Buffer(self.ctx, mf.READ_ONLY, d2c.size * 4)

        shape = np.array([zrange, xmax], dtype=np.int32)

        # we need to make sure this is contiguous
        s2c_ = np.ascontiguousarray(s2c.ravel().astype(np.float32))
        cl.enqueue_copy(self.queue, s2c_buf, s2c_)

        c2d_ = np.ascontiguousarray(c2d.ravel().astype(np.float32))
        cl.enqueue_copy(self.queue, c2d_buf, c2d_)

        d2c_ = np.ascontiguousarray(d2c.ravel().astype(np.float32))
        cl.enqueue_copy(self.queue, d2c_buf, d2c_)

        # I fill with zeros as I add to initial buffer (makes handling the 2 bs easier)
        cl.enqueue_fill_buffer(self.queue, self.sxz_buf, np.float32(0.0), 0, zrange * xmax * 4)
        cl.enqueue_fill_buffer(self.queue, out_buf_2, np.float32(0.0), 0, zrange * xmax * 4)

        # start of the actual calculations bits

        # note that this is eps also defined in the kernels
        eps = 0.000000000001
        do_second_stuff = np.sum(np.abs(b2)) >= eps

        self.displace_r(shape, self.sxz_buf, s2c_buf, c2d_buf, d2c_buf, p1, u, g, b1, c2d, nu, dt)

        if do_second_stuff:
            self.displace_r(shape, self.sxz_buf, s2c_buf, c2d_buf, d2c_buf, p2, u, g, b2, c2d, nu, dt)

        # -dz as we -p1 so becomes +dz
        self.displace_r(shape, out_buf_2, s2c_buf, c2d_buf, d2c_buf, p1-dz, u, g, b1, c2d, nu, dt)

        if do_second_stuff:
            self.displace_r(shape, out_buf_2, s2c_buf, c2d_buf, d2c_buf, p2-dz, u, g, b2, c2d, nu, dt)

        dz_32 = np.float32(dz[2])
        self.disp_r_prog.difference(self.queue, shape, None, self.sxz_buf, out_buf_2, dz_32)

    def displace_r(self, shape, out_buf, s2c_buf, c2d_buf, d2c_buf, p, u, g, b, c2d, nu, dt):

        b_screw = np.dot(b, u)
        b_edge = b - b_screw * u  # NB a vector
        b_edge_d = c2d @ b_edge
        b_unit = b_edge / (np.dot(b_edge, b_edge) ** 0.5)

        # float3 is actually a float4 in disguise?
        nu_32 = np.float32(nu)
        dt_32 = np.float32(dt)
        b_screw_32 = np.float32(b_screw)
        p1_32 = np.append(p, 0.0).astype(np.float32)
        u_32 = np.append(u, 0.0).astype(np.float32)
        g_32 = np.append(g, 0.0).astype(np.float32)
        b_32 = np.append(b, 0.0).astype(np.float32)
        b_unit_32 = np.append(b_unit, 0.0).astype(np.float32)
        b_edge_d_32 = np.append(b_edge_d, 0.0).astype(np.float32)

        self.disp_r_prog.displacement(self.queue, shape, None, out_buf, s2c_buf, c2d_buf, d2c_buf, p1_32, u_32, g_32, b_32, b_unit_32, b_edge_d_32, b_screw_32, dt_32, nu_32)

    def get_sxz_buffer(self, xmax, zrange):
        output = np.zeros((zrange, xmax), dtype=np.float32)
        cl.enqueue_copy(self.queue, output, self.sxz_buf)

        return output

    def calculate_image(self, xmax, ymax, zmax, zrange, hpad, dt, phi, s, firs, las, Xg, X0i, g, b1):
        if self.sxz_buf is None:
            raise Exception('sxz buffer has not been created. Did you run calculate_displacements?')

        mf = cl.mem_flags
        in_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, ymax * xmax * 4 * 4)

        #

        shape = np.array([xmax, ymax], dtype=np.int32)

        # generate our F values
        tableau = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        full = np.tile(tableau, xmax*ymax)
        cl.enqueue_copy(self.queue, in_out_buf, full.astype(np.float32))

        g_32 = np.append(g, 0.0).astype(np.float32)
        b_32 = np.append(b1, 0.0).astype(np.float32)

        x_g_32 = np.complex64(Xg)  # complex64 as it is 2 32-bit numbers

        z_range_32 = np.float32(zrange)
        dt_32 = np.float32(dt)
        phi_32 = np.float32(phi)
        firs_32 = np.float32(firs)
        las_32 = np.float32(las)
        s_32 = np.float32(s)
        x_0_i_32 = np.float32(X0i)

        h_pad_i32 = np.int32(hpad)
        z_max_i32 = np.int32(zmax)

        for k in range(zmax):
            k_i32 = np.int32(k)

            self.image_prog.calculate_f(self.queue, shape, None, in_out_buf, self.sxz_buf, g_32, b_32, x_g_32, x_0_i_32, z_range_32, dt_32, phi_32, firs_32, las_32, s_32, k_i32, h_pad_i32, z_max_i32)

        self.image_prog.calculate_image(self.queue, shape, None, in_out_buf)

        output = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, output, in_out_buf)

        image_bf = np.flip(output[:, :, 0], axis=0)
        image_df = np.flip(output[:, :, 2], axis=0)

        # image_bf = output[:, :, 0]
        # image_df = output[:, :, 2]

        return image_bf, image_df
