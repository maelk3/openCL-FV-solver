
// We use openCL image3d_t and openGL 3d textures to represents the
// Euler state vector over a 2d rectangle. The first two dimensions
// represent the width and height (plus padding with `ghost` regions
// to treat the boundary conditions) and the last one is the different
// components of the state vector Q: mass density, 2d momentum vector
// and energy. In order to have an trivially parallel code that does
// not rely on branches, the treatement of boundary conditions has to
// be done this way: the physical domain corresponds only to the inner
// rectangle in the picture below and large borders have to be copied
// from the physical domain to mimic different boundary conditions:
// periodic in our case. Furthermore, these read-only ghost cells need
// to be multiple cells thick as our computational stencil is rather
// large (5x5 stencil for the 5th order WENO reconstruction). We chose
// the size of the local work group in order to be efficient.
//
//      LOCAL_WORK_SIZE_X
//            ┌┈┈┴┈┈┐
//            ╭───────────────────────────╮
//            │  ghost replicated cells   │ ╮
//            │     ╭───────────────╮     │ │ ╮
//            │     │               │     │ │ │ ╮
//            │     │ computational │     │ │ │ │
//            │     │    domain     │     │ │ │ │
//            │     │               │     │ │ │ │
//            │     │               │     │ │ │ │
//            │     ╰───────────────╯     │ │ │ │
//            │                           │ │ │ │
//            ╰───────────────────────────╯ │ │ │
//             ╰────────────────────────────╯ │ │
//              ╰─────────────────────────────╯ │
//               ╰──────────────────────────────╯
//
// The position of the different unknown quantities Q and their
// reconstruction Q_{l,r,d,u} are laid out according to the next two
// pictures:
//                       Q_u(i,j+1)
//               ╭─────────────────────────╮
//               │        Q_d(i,j)         │
//               │                         │
//               │                         │
//               │                         │
//     Q_l(i-1,j)│Q_r(i,j)  Q(i,j) Q_l(i,j)│Q_r(i+1,j)
//               │                         │
//               │                         │
//               │                         │
//               │         Q_u(i,j)        │
//               ╰─────────────────────────╯
//                        Q_d(i,j-1)
//
//                        ────┬────
//                            │
//                            │
//                            │
//                         F(i,j)
//                  Q_l(i-1,j)│Q_r(i,j)
//                            │
//                            │
//                        ────┴────

#define GAMMA 1.4f

// Initialize the state vector for the Euler conservation law
__kernel void init_Q(__write_only image3d_t Q) {
  size_t g_id_x = get_global_id(0);
  size_t g_id_y = get_global_id(1);

  size_t w_size_x = get_local_size(0);
  size_t w_size_y = get_local_size(1);

  size_t offset_x = get_global_offset(0);
  size_t offset_y = get_global_offset(1);

  size_t width = get_global_size(0);
  size_t height = get_global_size(1);

  float2 pos = (float2)(((float)(g_id_x-w_size_x))/((float)width),
			((float)(g_id_y-w_size_y))/((float)height)) - (float2)(0.5f);

  float sigma = 0.05f/sqrt(2.0f);

  float rho   = fabs(pos.y) < 0.25 ? 2.0f : 1.0f;
  float rho_u = rho*(fabs(pos.y) < 0.25 ? -0.5f : 0.5f);
  float rho_v = rho*(0.1*sin(4.0f*M_PI_F*pos.x)*(exp(-((pos.y-0.25f)/(2*sigma))*((pos.y-0.25f)/(2*sigma)))+exp(-((pos.y+0.25f)/(2*sigma))*((pos.y+0.25f)/(2*sigma)))));
  float e     = 2.0/(GAMMA-1.0f) + 0.5*(rho_u*rho_u+rho_v*rho_v)/rho;

  write_imagef(Q, (int4)(g_id_x,g_id_y,0,0), (float4)(rho,0.0f,0.0f,0.0f));
  write_imagef(Q, (int4)(g_id_x,g_id_y,1,0), (float4)(rho_u,0.0f,0.0f,0.0f));
  write_imagef(Q, (int4)(g_id_x,g_id_y,2,0), (float4)(rho_v,0.0f,0.0f,0.0f));
  write_imagef(Q, (int4)(g_id_x,g_id_y,3,0), (float4)(e,0.0f,0.0f,0.0f));
}

// computes min and max values on a 2d slice of a 3d image per work
// group and write them onto the buffers `{min,max}_values`, one float
// per work group
__kernel void minmax_image3d_slice(__read_only image3d_t image,
				   __local float* l_min,
				   __local float* l_max,
				   __global float* min_values,
				   __global float* max_values) {
  size_t l_id_x = get_local_id(0);
  size_t l_id_y = get_local_id(1);

  size_t w_size_x = get_local_size(0);
  size_t w_size_y = get_local_size(1);

  size_t g_id_x = get_global_id(0);
  size_t g_id_y = get_global_id(1);
  size_t g_id_z = get_global_id(2);

  l_max[l_id_x+l_id_y*w_size_x] = read_imagef(image, (int4)(g_id_x, g_id_y, g_id_z, 0)).x;
  l_min[l_id_x+l_id_y*w_size_x] = read_imagef(image, (int4)(g_id_x, g_id_y, g_id_z, 0)).x;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int k=w_size_x/2; k > 0; k >>= 1) {
    if(l_id_x < k) {
      l_max[l_id_x+l_id_y*w_size_x] = max(l_max[l_id_x+l_id_y*w_size_x], l_max[l_id_x+k+l_id_y*w_size_x]);
      l_min[l_id_x+l_id_y*w_size_x] = min(l_min[l_id_x+l_id_y*w_size_x], l_min[l_id_x+k+l_id_y*w_size_x]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

    for(int k=w_size_y/2; k > 0; k >>= 1) {
      if(l_id_x == 0 && l_id_y < k) {
	l_max[l_id_y*w_size_x] = max(l_max[l_id_y*w_size_x], l_max[(l_id_y+k)*w_size_x]);
	l_min[l_id_y*w_size_x] = min(l_min[l_id_y*w_size_x], l_min[(l_id_y+k)*w_size_x]);
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id_x == 0 && l_id_y == 0) {
      max_values[get_group_id(0)+get_group_id(1)*get_num_groups(0)] = l_max[0];
      min_values[get_group_id(0)+get_group_id(1)*get_num_groups(0)] = l_min[0];
    }
}

__kernel void max_image2d(__read_only image2d_t image,
			  __local float* l_max,
			  __global float* max_values) {
  size_t l_id_x = get_local_id(0);
  size_t l_id_y = get_local_id(1);

  size_t w_size_x = get_local_size(0);
  size_t w_size_y = get_local_size(1);

  size_t g_id_x = get_global_id(0);
  size_t g_id_y = get_global_id(1);

  l_max[l_id_x+l_id_y*w_size_x] = read_imagef(image, (int2)(g_id_x, g_id_y)).x;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int k=w_size_x/2; k > 0; k >>= 1) {
    if(l_id_x < k)
      l_max[l_id_x+l_id_y*w_size_x] = max(l_max[l_id_x+l_id_y*w_size_x], l_max[l_id_x+k+l_id_y*w_size_x]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

    for(int k=w_size_y/2; k > 0; k >>= 1) {
      if(l_id_x == 0 && l_id_y < k)
	l_max[l_id_y*w_size_x] = max(l_max[l_id_y*w_size_x], l_max[(l_id_y+k)*w_size_x]);
    barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id_x == 0 && l_id_y == 0)
      max_values[get_group_id(0)+get_group_id(1)*get_num_groups(0)] = l_max[0];
}

__kernel void max_buffer(__global float* input_max,
			 __local float* l_max,
			 __global float* max_values) {
  size_t g_id = get_global_id(0);
  size_t l_id = get_local_id(0);
  size_t w_size = get_local_size(0);

  l_max[l_id] = input_max[g_id];
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int k=w_size/2; k > 0; k >>= 1) {
    if(l_id < k) {
      l_max[l_id] = max(l_max[l_id], l_max[l_id+k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(l_id == 0) {
      max_values[get_group_id(0)] = l_max[0];
  }
}

// computes min and max values of `input_{min,max}` per work group and
// write them onto the buffers `{min,max}_values`, one float per work
// group
__kernel void minmax_buffer(__global float* input_min,
			    __global float* input_max,
			    __local float* l_min,
			    __local float* l_max,
			    __global float* min_values,
			    __global float* max_values) {
  size_t g_id = get_global_id(0);
  size_t l_id = get_local_id(0);
  size_t w_size = get_local_size(0);

  l_max[l_id] = input_max[g_id];
  l_min[l_id] = input_min[g_id];
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int k=w_size/2; k > 0; k >>= 1) {
    if(l_id < k) {
      l_max[l_id] = max(l_max[l_id], l_max[l_id+k]);
      l_min[l_id] = min(l_min[l_id], l_min[l_id+k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(l_id == 0) {
      max_values[get_group_id(0)] = l_max[0];
      min_values[get_group_id(0)] = l_min[0];
  }
}

__kernel void set_periodic_boundary_conditions_x(__read_write image3d_t image) {
  size_t g_id_x = get_global_id(0);
  size_t g_id_y = get_global_id(1);

  size_t w_size_x = get_local_size(0);
  size_t w_size_y = get_local_size(1);

  size_t n_x = get_image_width(image);

  for(int k=0; k<4; k++) {
    write_imagef(image, (int4)(n_x-w_size_x+g_id_x, w_size_y+g_id_y, k, 0), read_imagef(image, (int4)(w_size_x+g_id_x, w_size_y+g_id_y, k, 0)));
    write_imagef(image, (int4)(g_id_x, w_size_y+g_id_y, k, 0), read_imagef(image, (int4)(n_x-2*w_size_x+g_id_x, w_size_y+g_id_y, k, 0)));
  }
}

__kernel void set_periodic_boundary_conditions_y(__read_write image3d_t image) {
  size_t g_id_x = get_global_id(0);
  size_t g_id_y = get_global_id(1);

  size_t w_size_x = get_local_size(0);
  size_t w_size_y = get_local_size(1);

  size_t n_y = get_image_height(image);

  for(int k=0; k<4; k++) {
    write_imagef(image, (int4)(g_id_x, g_id_y, k, 0), read_imagef(image, (int4)(g_id_x, n_y-2*w_size_y+g_id_y, k, 0)));
    write_imagef(image, (int4)(g_id_x, n_y-w_size_y+g_id_y, k, 0), read_imagef(image, (int4)(g_id_x, w_size_y+g_id_y, k, 0)));
  }
}

#define Q(i,j,k) read_imagef(Q, (int4)((i), (j), (k), 0)).x

__kernel void compute_weno_reconstruction(__read_only image3d_t Q,
					  __write_only image3d_t Q_l,
					  __write_only image3d_t Q_r,
					  __write_only image3d_t Q_d,
					  __write_only image3d_t Q_u) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k = get_global_id(2);

  float beta_0_x = 13.0f/12.0f*(Q(i-2,j,k)-2*Q(i-1,j,k)+Q(i,j,k))*(Q(i-2,j,k)-2*Q(i-1,j,k)+Q(i,j,k)) +
    0.25f*(Q(i-2,j,k)-4.0f*Q(i-1,j,k)+3.0f*Q(i,j,k))*(Q(i-2,j,k)-4.0f*Q(i-1,j,k)+3.0f*Q(i,j,k));
  float beta_1_x = 13.0f/12.0f*(Q(i-1,j,k)-2.0f*Q(i,j,k)+Q(i+1,j,k))*(Q(i-1,j,k)-2.0f*Q(i,j,k)+Q(i+1,j,k)) +
    0.25f*(Q(i-1,j,k)-Q(i+1,j,k))*(Q(i-1,j,k)-Q(i+1,j,k));
  float beta_2_x = 13.0f/12.0f*(Q(i,j,k)-2.0f*Q(i+1,j,k)+Q(i+2,j,k))*(Q(i,j,k)-2.0f*Q(i+1,j,k)+Q(i+2,j,k)) +
    0.25f*(3.0f*Q(i,j,k)-4.0f*Q(i+1,j,k)+Q(i+2,j,k))*(3.0f*Q(i,j,k)-4.0f*Q(i+1,j,k)+Q(i+2,j,k));

  const float epsilon = 0.000001f;
  float alpha_l_0 = 0.1/((beta_0_x+epsilon)*(beta_0_x+epsilon));
  float alpha_l_1 =  0.6/((beta_1_x+epsilon)*(beta_1_x+epsilon));
  float alpha_l_2 = 0.3/((beta_2_x+epsilon)*(beta_2_x+epsilon));

  float alpha_r_0 = 0.3/((beta_0_x+epsilon)*(beta_0_x+epsilon));
  float alpha_r_1 =  0.6/((beta_1_x+epsilon)*(beta_1_x+epsilon));
  float alpha_r_2 = 0.1/((beta_2_x+epsilon)*(beta_2_x+epsilon));

  float sum_l = alpha_l_0+alpha_l_1+alpha_l_2;
  float sum_r = alpha_r_0+alpha_r_1+alpha_r_2;

  float w_l_0 = alpha_l_0/sum_l;
  float w_l_1 = alpha_l_1/sum_l;
  float w_l_2 = alpha_l_2/sum_l;

  float w_r_0 = alpha_r_0/sum_r;
  float w_r_1 = alpha_r_1/sum_r;
  float w_r_2 = alpha_r_2/sum_r;

  write_imagef(Q_l, (int4)(i,j,k,0), w_l_0*(1.0f/3.0f*Q(i-2,j,k)-7.0f/6.0f*Q(i-1,j,k)+11.0f/6.0f*Q(i,j,k)) +
	       w_l_1*(-1.0f/6.0f*Q(i-1,j,k)+5.0f/6.0f*Q(i,j,k)+1.0f/3.0f*Q(i+1,j,k)) +
	       w_l_2*(1.0f/3.0f*Q(i,j,k)+5.0f/6.0f*Q(i+1,j,k)-1.0f/6.0f*Q(i+2,j,k)));

  write_imagef(Q_r, (int4)(i,j,k,0), w_r_0*(1.0f/3.0f*Q(i,j,k)+5.0f/6.0f*Q(i-1,j,k)-1.0f/6.0f*Q(i-2,j,k)) +
	       w_r_1*(-1.0f/6.0f*Q(i+1,j,k)+5.0f/6.0f*Q(i,j,k)+1.0f/3.0f*Q(i-1,j,k)) +
	       w_r_2*(1.0f/3.0f*Q(i+2,j,k)-7.0f/6.0f*Q(i+1,j,k)+11.0f/6.0f*Q(i,j,k)));

  float beta_0_y = 13.0f/12.0f*(Q(i,j-2,k)-2.0f*Q(i,j-1,k)+Q(i,j,k))*(Q(i,j-2,k)-2.0f*Q(i,j-1,k)+Q(i,j,k)) +
    0.25f*(Q(i,j-2,k)-4.0f*Q(i,j-1,k)+3.0f*Q(i,j,k))*(Q(i,j-2,k)-4.0f*Q(i,j-1,k)+3.0f*Q(i,j,k));
  float beta_1_y = 13.0f/12.0f*(Q(i,j-1,k)-2.0f*Q(i,j,k)+Q(i,j+1,k))*(Q(i,j-1,k)-2.0f*Q(i,j,k)+Q(i,j+1,k)) +
    0.25f*(Q(i,j-1,k)-Q(i,j+1,k))*(Q(i, j-1,k)-Q(i, j+1,k));
  float beta_2_y = 13.0f/12.0f*(Q(i,j,k)-2.0f*Q(i,j+1,k)+Q(i,j+2,k))*(Q(i,j,k)-2.0f*Q(i,j+1,k)+Q(i,j+2,k)) +
    0.25f*(3.0f*Q(i,j,k)-4.0f*Q(i,j+1,k)+Q(i,j+2,k))*(3.0f*Q(i,j,k)-4.0f*Q(i,j+1,k)+Q(i,j+2,k));

  float alpha_d_0 = 0.1/((beta_0_y+epsilon)*(beta_0_y+epsilon));
  float alpha_d_1 =  0.6/((beta_1_y+epsilon)*(beta_1_y+epsilon));
  float alpha_d_2 = 0.3/((beta_2_y+epsilon)*(beta_2_y+epsilon));

  float alpha_u_0 = 0.3/((beta_0_y+epsilon)*(beta_0_y+epsilon));
  float alpha_u_1 = 0.6/((beta_1_y+epsilon)*(beta_1_y+epsilon));
  float alpha_u_2 = 0.1/((beta_2_y+epsilon)*(beta_2_y+epsilon));

  float sum_d = alpha_d_0+alpha_d_1+alpha_d_2;
  float sum_u = alpha_u_0+alpha_u_1+alpha_u_2;

  float w_d_0 = alpha_d_0/sum_d;
  float w_d_1 = alpha_d_1/sum_d;
  float w_d_2 = alpha_d_2/sum_d;

  float w_u_0 = alpha_u_0/sum_u;
  float w_u_1 = alpha_u_1/sum_u;
  float w_u_2 = alpha_u_2/sum_u;

  write_imagef(Q_d, (int4)(i,j,k,0), w_d_0*(1.0f/3.0f*Q(i,j-2,k)-7.0f/6.0f*Q(i,j-1,k)+11.0f/6.0f*Q(i,j,k)) +
	       w_d_1*(-1.0f/6.0f*Q(i,j-1,k)+5.0f/6.0f*Q(i,j,k)+1.0f/3.0f*Q(i,j+1,k)) +
	       w_d_2*(1.0f/3.0f*Q(i,j,k)+5.0f/6.0f*Q(i,j+1,k)-1.0f/6.0f*Q(i,j+2,k)));

  write_imagef(Q_u, (int4)(i,j,k,0), w_u_0*(1.0f/3.0f*Q(i,j,k)+5.0f/6.0f*Q(i,j-1,k)-1.0f/6.0f*Q(i,j-2,k)) +
	       w_u_1*(-1.0f/6.0f*Q(i,j+1,k)+5.0f/6.0f*Q(i,j,k)+1.0f/3.0f*Q(i,j-1,k)) +
	       w_u_2*(1.0f/3.0f*Q(i,j+2,k)-7.0f/6.0f*Q(i,j+1,k)+11.0f/6.0f*Q(i,j,k)));
}

#define Q_l(i,j,k) read_imagef(Q_l, (int4)((i), (j), (k), 0)).x
#define Q_r(i,j,k) read_imagef(Q_r, (int4)((i), (j), (k), 0)).x
#define Q_u(i,j,k) read_imagef(Q_u, (int4)((i), (j), (k), 0)).x
#define Q_d(i,j,k) read_imagef(Q_d, (int4)((i), (j), (k), 0)).x

__kernel void compute_x_flux(__read_only image3d_t Q_l,
			     __read_only image3d_t Q_r,
			     __write_only image3d_t F,
			     __write_only image2d_t S_x) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);

  float rho_l   = Q_l(i-1,j,0);
  float rho_u_l = Q_l(i-1,j,1);
  float rho_v_l = Q_l(i-1,j,2);
  float rho_e_l = Q_l(i-1,j,3);
  float u_l = rho_u_l/rho_l;
  float v_l = rho_v_l/rho_l;
  float p_l = (GAMMA-1)*(rho_e_l-0.5*rho_l*(u_l*u_l+v_l*v_l));
  float H_l = (rho_e_l+p_l)/rho_l;
  float c_l = sqrt((GAMMA-1)*(H_l-0.5*(u_l*u_l+v_l*v_l)));

  float rho_r   = Q_r(i,j,0);
  float rho_u_r = Q_r(i,j,1);
  float rho_v_r = Q_r(i,j,2);
  float rho_e_r = Q_r(i,j,3);
  float u_r = rho_u_r/rho_r;
  float v_r = rho_v_r/rho_r;
  float p_r = (GAMMA-1)*(rho_e_r-0.5*rho_r*(u_r*u_r+v_r*v_r));
  float H_r = (rho_e_r+p_r)/rho_r;
  float c_r = sqrt((GAMMA-1)*(H_r-0.5*(u_r*u_r+v_r*v_r)));

  float sqrt_rho_l = sqrt(rho_l);
  float sqrt_rho_r = sqrt(rho_r);
  float sum_weights = sqrt_rho_l+sqrt_rho_r;

  float u_roe = (sqrt_rho_l*u_l+sqrt_rho_r*u_r)/sum_weights;
  float v_roe = (sqrt_rho_l*v_l+sqrt_rho_r*v_r)/sum_weights;
  float H_roe = (sqrt_rho_l*H_l+sqrt_rho_r*H_r)/sum_weights;
  float c_roe = sqrt((GAMMA-1)*(H_roe-0.5*(u_roe*u_roe+v_roe*v_roe)));

  float S_l = min(u_roe-c_roe, u_l-c_l);
  float S_r = max(u_roe+c_roe, u_r+c_r);

  write_imagef(S_x, (int2)(i,j), max(-S_l, S_r));

  float F_l[4] = {rho_u_l,
		  rho_u_l*rho_u_l/rho_l + p_l,
		  rho_u_l*v_l,
		  rho_u_l*H_l};

  float F_r[4] = {rho_u_r,
		  rho_u_r*rho_u_r/rho_r + p_r,
		  rho_u_r*v_r,
		  rho_u_r*H_r};

  float S_l_clamp = min(S_l, 0.0f);
  float S_r_clamp = max(S_r, 0.0f);
  for(size_t k=0; k<4; k++)
    write_imagef(F, (int4)(i,j,k,0), ((S_r_clamp*F_l[k]-S_l_clamp*F_r[k])+S_r_clamp*S_l_clamp*(Q_r(i,j,k)-Q_l(i-1,j,k)))/(S_r_clamp-S_l_clamp));
}

__kernel void compute_y_flux(__read_only image3d_t Q_d,
			     __read_only image3d_t Q_u,
			     __write_only image3d_t G,
			     __write_only image2d_t S_y) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);

  float rho_d   = Q_d(i,j-1,0);
  float rho_u_d = Q_d(i,j-1,1);
  float rho_v_d = Q_d(i,j-1,2);
  float rho_e_d = Q_d(i,j-1,3);
  float u_d = rho_u_d/rho_d;
  float v_d = rho_v_d/rho_d;
  float p_d = (GAMMA-1)*(rho_e_d-0.5*rho_d*(u_d*u_d+v_d*v_d));
  float H_d = (rho_e_d+p_d)/rho_d;
  float c_d = sqrt((GAMMA-1)*(H_d-0.5*(u_d*u_d+v_d*v_d)));

  float rho_u   = Q_u(i,j,0);
  float rho_u_u = Q_u(i,j,1);
  float rho_v_u = Q_u(i,j,2);
  float rho_e_u = Q_u(i,j,3);
  float u_u = rho_u_u/rho_u;
  float v_u = rho_v_u/rho_u;
  float p_u = (GAMMA-1)*(rho_e_u-0.5*rho_u*(u_u*u_u+v_u*v_u));
  float H_u = (rho_e_u+p_u)/rho_u;
  float c_u = sqrt((GAMMA-1)*(H_u-0.5*(u_u*u_u+v_u*v_u)));

  float sqrt_rho_d = sqrt(rho_d);
  float sqrt_rho_u = sqrt(rho_u);
  float sum_weights = sqrt_rho_d+sqrt_rho_u;

  float u_roe = (sqrt_rho_d*u_d+sqrt_rho_u*u_u)/sum_weights;
  float v_roe = (sqrt_rho_d*v_d+sqrt_rho_u*v_u)/sum_weights;
  float H_roe = (sqrt_rho_d*H_d+sqrt_rho_u*H_u)/sum_weights;
  float c_roe = sqrt((GAMMA-1)*(H_roe-0.5*(u_roe*u_roe+v_roe*v_roe)));

  float S_d = min(v_roe-c_roe, v_d-c_d);
  float S_u = max(v_roe+c_roe, v_u+c_u);

  write_imagef(S_y, (int2)(i,j), max(-S_d, S_u));

  float G_d[4] = {rho_v_d,
		  rho_v_d*u_d,
		  rho_v_d*rho_v_d/rho_d + p_d,
		  rho_v_d*H_d};

  float G_u[4] = {rho_v_u,
		  rho_v_u*u_u,
		  rho_v_u*rho_v_u/rho_u + p_u,
		  rho_v_u*H_u};

  float S_d_clamp = min(S_d, 0.0f);
  float S_u_clamp = max(S_u, 0.0f);
  for(size_t k=0; k<4; k++)
    write_imagef(G, (int4)(i,j,k,0), ((S_u_clamp*G_d[k]-S_d_clamp*G_u[k])+S_u_clamp*S_d_clamp*(Q_u(i,j,k)-Q_d(i,j-1,k)))/(S_u_clamp-S_d_clamp));
}

#define F(i,j,k) read_imagef(F, (int4)((i), (j), (k), 0)).x
#define G(i,j,k) read_imagef(G, (int4)((i), (j), (k), 0)).x
#define Q_prev(i,j,k) read_imagef(Q_prev, (int4)((i), (j), (k), 0)).x

__kernel void set_Q_1(__write_only image3d_t Q_1,
		      __read_only image3d_t Q_prev,
		      __read_only image3d_t F,
		      __read_only image3d_t G,
		      float delta_x,
		      float delta_y,
		      float delta_t) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k = get_global_id(2);

  write_imagef(Q_1, (int4)(i,j,k,0), Q_prev(i,j,k) - delta_t/delta_x*(F(i+1,j,k)-F(i,j,k)) - delta_t/delta_y*(G(i,j+1,k)-G(i,j,k)));
}

#define Q_1(i,j,k) read_imagef(Q_1, (int4)((i), (j), (k), 0)).x

__kernel void set_Q_2(__write_only image3d_t Q_2,
		      __read_only image3d_t Q_prev,
		      __read_only image3d_t Q_1,
		      __read_only image3d_t F,
		      __read_only image3d_t G,
		      float delta_x,
		      float delta_y,
		      float delta_t) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k = get_global_id(2);

  write_imagef(Q_2, (int4)(i,j,k,0), 0.75f*Q_prev(i,j,k) + 0.25f*Q_1(i,j,k) +0.25f*(-delta_t/delta_x*(F(i+1,j,k)-F(i,j,k)) - delta_t/delta_y*(G(i,j+1,k)-G(i,j,k))));
}

#define Q_2(i,j,k) read_imagef(Q_2, (int4)((i), (j), (k), 0)).x

__kernel void set_Q_next(__write_only image3d_t Q_next,
			 __read_only image3d_t Q_prev,
			 __read_only image3d_t Q_1,
			 __read_only image3d_t Q_2,
			 __read_only image3d_t F,
			 __read_only image3d_t G,
			 float delta_x,
			 float delta_y,
			 float delta_t) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t k = get_global_id(2);

  write_imagef(Q_next, (int4)(i,j,k,0), 1.0f/3.0f*Q_prev(i,j,k) + 2.0f/3.0f*Q_2(i,j,k) + 2.0f/3.0f*(-delta_t/delta_x*(F(i+1,j,k)-F(i,j,k)) - delta_t/delta_y*(G(i,j+1,k)-G(i,j,k))));
}

__kernel void copy_image_3d_to_2d(__read_only image3d_t image_src,
				  __write_only image2d_t image_dest) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);

  size_t offset_x = get_global_offset(0);
  size_t offset_y = get_global_offset(1);

  float4 color = (float4)(read_imagef(image_src, (int4)(i,j,0,0)).x,
			  read_imagef(image_src, (int4)(i,j,1,0)).x,
			  read_imagef(image_src, (int4)(i,j,2,0)).x,
			  read_imagef(image_src, (int4)(i,j,3,0)).x);

  write_imagef(image_dest, (int2)(i-offset_x,j-offset_y), color);
}
