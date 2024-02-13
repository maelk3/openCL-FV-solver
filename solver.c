#include "global.h"
#include "solver.h"
#include "utilities.h"

#include <CL/cl.h>
#include <CL/cl_gl.h>

typedef struct solver_context_t {
  cl_context context;
  cl_device_id device_id;
  cl_command_queue queue;
  float rho_min, rho_max;
  float t;
  float delta_x;
  float delta_y;
  float delta_t;
  cl_program program;
  struct {
    cl_mem Q_prev, Q_next, Q_1, Q_2;
    cl_mem Q_l, Q_r, Q_u, Q_d;
    cl_mem F, G;
    cl_mem S_x, S_y;
    cl_mem min, max;
  } buffers;
  struct {
    cl_kernel init_Q;
    cl_kernel minmax_image3d_slice;
    cl_kernel minmax_buffer;
    cl_kernel max_image2d;
    cl_kernel max_buffer;
    cl_kernel set_periodic_boundary_conditions_x;
    cl_kernel set_periodic_boundary_conditions_y;
    cl_kernel compute_weno_reconstruction;
    cl_kernel compute_x_flux;
    cl_kernel compute_y_flux;
    cl_kernel set_Q_1;
    cl_kernel set_Q_2;
    cl_kernel set_Q_next;
  } kernels;
} solver_context_t;

/** callback function by the openCL driver on context errors */
static void cl_context_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data);

#ifdef DEBUG
/** returns a static error string corresponding to the openCL error
 * code */
static const char* get_error_string(cl_int error);
#endif

/** Creates an openCL program object associated with `context` and
 * builds the source file in `filepath` for device `device_id` with
 * build options `build_options` */
static cl_program build_program_from_source(const char* filepath, const char* build_options, cl_context context, cl_device_id device_id);

/** computes the range of values taken by a slice of the 3d image
 * `ctx->buffers.Q_init` representing the density and stores it in the
 * context. Note that clEnqueueAcquireGLObjects must be called ahead
 * of tile to be able to read the shared texture between openGL and
 * openCL */
static void solver_compute_density_range(solver_context_t* ctx);

/** Apply periodic boundary conditions to the openCL image `image` */
static void solver_set_periodic_boundary_conditions(solver_context_t* ctx, cl_mem image);

/** returns the max value of the inner cells (not including the ghost
 * cells) of the 2d image */
static float solver_get_max_value_image2d(solver_context_t* ctx, cl_mem image);

/** computes the weno reconstruction of the Euler system of variables Q
 * and computes the associated fluxes */
static void solver_compute_weno_reconstruction_and_fluxes(solver_context_t* ctx, cl_mem Q);

/** The macro OPENCL_CATCH_ERROR prints to stderr the opencl error name
 * with the current line number and file name and terminates the
 * program with EXIT_FAILURE in DEBUG mode. In release mode, this
 * macro executes the argument and discards the opencl error code. */
#ifdef NDEBUG
#define OPENCL_CATCH_ERROR(x) \
  do { (void) (x); } while (0)
#else
#define OPENCL_CATCH_ERROR(x) do {				\
    cl_int __err_code = (x);					\
    if(__err_code != 0) {					\
      fprintf(stderr, "OpenCL error %s caught in %s:%d\n",	\
	      get_error_string(x),				\
	      __FILE__,						\
	      __LINE__);					\
      exit(EXIT_FAILURE);					\
    }								\
  } while (0)

#define CASE_RETURN_STRING(x) case x: return #x;

/** `get_error_string' return a static string associated with the
 * openCL error code returned by any openCL function */
static const char* get_error_string(cl_int error) {
  switch (error) {
    CASE_RETURN_STRING(CL_SUCCESS)
      CASE_RETURN_STRING(CL_DEVICE_NOT_FOUND)
      CASE_RETURN_STRING(CL_DEVICE_NOT_AVAILABLE)
      CASE_RETURN_STRING(CL_COMPILER_NOT_AVAILABLE)
      CASE_RETURN_STRING(CL_MEM_OBJECT_ALLOCATION_FAILURE)
      CASE_RETURN_STRING(CL_OUT_OF_RESOURCES)
      CASE_RETURN_STRING(CL_OUT_OF_HOST_MEMORY)
      CASE_RETURN_STRING(CL_PROFILING_INFO_NOT_AVAILABLE)
      CASE_RETURN_STRING(CL_MEM_COPY_OVERLAP)
      CASE_RETURN_STRING(CL_IMAGE_FORMAT_MISMATCH)
      CASE_RETURN_STRING(CL_IMAGE_FORMAT_NOT_SUPPORTED)
      CASE_RETURN_STRING(CL_BUILD_PROGRAM_FAILURE)
      CASE_RETURN_STRING(CL_MAP_FAILURE)
      CASE_RETURN_STRING(CL_MISALIGNED_SUB_BUFFER_OFFSET)
      CASE_RETURN_STRING(CL_COMPILE_PROGRAM_FAILURE)
      CASE_RETURN_STRING(CL_LINKER_NOT_AVAILABLE)
      CASE_RETURN_STRING(CL_LINK_PROGRAM_FAILURE)
      CASE_RETURN_STRING(CL_DEVICE_PARTITION_FAILED)
      CASE_RETURN_STRING(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
      CASE_RETURN_STRING(CL_INVALID_VALUE)
      CASE_RETURN_STRING(CL_INVALID_DEVICE_TYPE)
      CASE_RETURN_STRING(CL_INVALID_PLATFORM)
      CASE_RETURN_STRING(CL_INVALID_DEVICE)
      CASE_RETURN_STRING(CL_INVALID_CONTEXT)
      CASE_RETURN_STRING(CL_INVALID_QUEUE_PROPERTIES)
      CASE_RETURN_STRING(CL_INVALID_COMMAND_QUEUE)
      CASE_RETURN_STRING(CL_INVALID_HOST_PTR)
      CASE_RETURN_STRING(CL_INVALID_MEM_OBJECT)
      CASE_RETURN_STRING(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
      CASE_RETURN_STRING(CL_INVALID_IMAGE_SIZE)
      CASE_RETURN_STRING(CL_INVALID_SAMPLER)
      CASE_RETURN_STRING(CL_INVALID_BINARY)
      CASE_RETURN_STRING(CL_INVALID_BUILD_OPTIONS)
      CASE_RETURN_STRING(CL_INVALID_PROGRAM)
      CASE_RETURN_STRING(CL_INVALID_PROGRAM_EXECUTABLE)
      CASE_RETURN_STRING(CL_INVALID_KERNEL_NAME)
      CASE_RETURN_STRING(CL_INVALID_KERNEL_DEFINITION)
      CASE_RETURN_STRING(CL_INVALID_KERNEL)
      CASE_RETURN_STRING(CL_INVALID_ARG_INDEX)
      CASE_RETURN_STRING(CL_INVALID_ARG_VALUE)
      CASE_RETURN_STRING(CL_INVALID_ARG_SIZE)
      CASE_RETURN_STRING(CL_INVALID_KERNEL_ARGS)
      CASE_RETURN_STRING(CL_INVALID_WORK_DIMENSION)
      CASE_RETURN_STRING(CL_INVALID_WORK_GROUP_SIZE)
      CASE_RETURN_STRING(CL_INVALID_WORK_ITEM_SIZE)
      CASE_RETURN_STRING(CL_INVALID_GLOBAL_OFFSET)
      CASE_RETURN_STRING(CL_INVALID_EVENT_WAIT_LIST)
      CASE_RETURN_STRING(CL_INVALID_EVENT)
      CASE_RETURN_STRING(CL_INVALID_OPERATION)
      CASE_RETURN_STRING(CL_INVALID_GL_OBJECT)
      CASE_RETURN_STRING(CL_INVALID_BUFFER_SIZE)
      CASE_RETURN_STRING(CL_INVALID_MIP_LEVEL)
      CASE_RETURN_STRING(CL_INVALID_GLOBAL_WORK_SIZE)
      CASE_RETURN_STRING(CL_INVALID_PROPERTY)
      CASE_RETURN_STRING(CL_INVALID_IMAGE_DESCRIPTOR)
      CASE_RETURN_STRING(CL_INVALID_COMPILER_OPTIONS)
      CASE_RETURN_STRING(CL_INVALID_LINKER_OPTIONS)
      CASE_RETURN_STRING(CL_INVALID_DEVICE_PARTITION_COUNT)
  default: return "CL_UNKNOWN_ERROR";
  }
}
#endif // NDEBUG

solver_context_t* solver_init(preview_context_t* preview_ctx) {
  solver_context_t* solver_ctx = malloc(sizeof(*solver_ctx));
  solver_ctx->t = 0;
  solver_ctx->delta_x = DELTA_X;
  solver_ctx->delta_y = DELTA_Y;

  // TODO change code to choose GPU device and platform to check for
  // cl_khr_gl_sharing extension
  cl_uint num_platforms;
  clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id* platform_ids = malloc(sizeof(cl_platform_id)*num_platforms);
  clGetPlatformIDs(num_platforms, platform_ids, NULL);

  cl_platform_id platform_id = platform_ids[0];

  cl_context_properties context_properties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id,
    CL_GL_CONTEXT_KHR, (cl_context_properties) previewer_get_gl_context(preview_ctx),
    CL_GLX_DISPLAY_KHR, (cl_context_properties) previewer_get_display(preview_ctx),
    0};

  free(platform_ids);

  size_t platform_name_size;
  clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &platform_name_size);
  char* platform_name = malloc(platform_name_size);
  clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);
  printf("platform name: %s\n", platform_name);
  free(platform_name);

  cl_int error_code = CL_SUCCESS;
  solver_ctx->context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, cl_context_callback, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);

  cl_uint num_devices;
  OPENCL_CATCH_ERROR(clGetContextInfo(solver_ctx->context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL));

  cl_device_id* device_ids = malloc(sizeof(cl_device_id)*num_devices);
  OPENCL_CATCH_ERROR(clGetContextInfo(solver_ctx->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id)*num_devices, device_ids, NULL));

  solver_ctx->device_id = device_ids[0];

  cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

  solver_ctx->queue = clCreateCommandQueueWithProperties(solver_ctx->context,
							 solver_ctx->device_id,
							 queue_properties,
							 &error_code);
  OPENCL_CATCH_ERROR(error_code);

  solver_ctx->program = build_program_from_source("./kernels/kernels.cl", "-cl-std=CL2.0", solver_ctx->context, solver_ctx->device_id);

  solver_ctx->buffers.Q_next = clCreateFromGLTexture(solver_ctx->context,
						     CL_MEM_READ_WRITE,
						     GL_TEXTURE_3D,
						     0,
						     previewer_get_texture(preview_ctx),
						     &error_code);
  OPENCL_CATCH_ERROR(error_code);

  cl_image_format image_format = {
    .image_channel_order = CL_R,
    .image_channel_data_type = CL_FLOAT,
};

  cl_image_desc image_desc_3D_image = {
    .image_type = CL_MEM_OBJECT_IMAGE3D,
    .image_width = WIDTH,
    .image_height = HEIGHT,
    .image_depth = 4,
  };

  // create all the openCL image objects and buffers
  solver_ctx->buffers.Q_prev = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.Q_1 = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.Q_2 = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.Q_l = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.Q_d = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.Q_u = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.Q_r = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.F   = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.G   = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_3D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);

  cl_image_desc image_desc_2D_image = {
    .image_type = CL_MEM_OBJECT_IMAGE2D,
    .image_width = WIDTH,
    .image_height = HEIGHT,
  };
  solver_ctx->buffers.S_x = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_2D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.S_y = clCreateImage(solver_ctx->context, CL_MEM_READ_WRITE, &image_format, &image_desc_2D_image, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);

  solver_ctx->buffers.min = clCreateBuffer(solver_ctx->context, CL_MEM_WRITE_ONLY, BUFFER_SIZE, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->buffers.max = clCreateBuffer(solver_ctx->context, CL_MEM_WRITE_ONLY, BUFFER_SIZE, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);

  // create all the openCL kernel objects
  solver_ctx->kernels.minmax_image3d_slice = clCreateKernel(solver_ctx->program, "minmax_image3d_slice", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.minmax_buffer = clCreateKernel(solver_ctx->program, "minmax_buffer", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.max_image2d = clCreateKernel(solver_ctx->program, "max_image2d", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.max_buffer = clCreateKernel(solver_ctx->program, "max_buffer", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.set_periodic_boundary_conditions_x = clCreateKernel(solver_ctx->program, "set_periodic_boundary_conditions_x", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.set_periodic_boundary_conditions_y = clCreateKernel(solver_ctx->program, "set_periodic_boundary_conditions_y", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.compute_weno_reconstruction = clCreateKernel(solver_ctx->program, "compute_weno_reconstruction", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.compute_x_flux = clCreateKernel(solver_ctx->program, "compute_x_flux", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.compute_y_flux = clCreateKernel(solver_ctx->program, "compute_y_flux", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.init_Q = clCreateKernel(solver_ctx->program, "init_Q", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.set_Q_1 = clCreateKernel(solver_ctx->program, "set_Q_1", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.set_Q_2 = clCreateKernel(solver_ctx->program, "set_Q_2", &error_code);
  OPENCL_CATCH_ERROR(error_code);
  solver_ctx->kernels.set_Q_next = clCreateKernel(solver_ctx->program, "set_Q_next", &error_code);
  OPENCL_CATCH_ERROR(error_code);

  // initialize the image with the initial conditions
  clEnqueueAcquireGLObjects(solver_ctx->queue, 1, &solver_ctx->buffers.Q_next, 0, NULL, NULL);

  const size_t global_work_offset[] = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y};
  const size_t global_work_size[]   = {WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y};
  const size_t local_work_size[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y};
  OPENCL_CATCH_ERROR(clSetKernelArg(solver_ctx->kernels.init_Q, 0, sizeof(cl_mem), &solver_ctx->buffers.Q_next));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(solver_ctx->queue,
					    solver_ctx->kernels.init_Q,
					    2,
					    global_work_offset,
					    global_work_size,
					    local_work_size,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(solver_ctx, solver_ctx->buffers.Q_next);

  solver_compute_density_range(solver_ctx);

  clEnqueueReleaseGLObjects(solver_ctx->queue, 1, &solver_ctx->buffers.Q_next, 0, NULL, NULL);
  OPENCL_CATCH_ERROR(error_code);
  clFinish(solver_ctx->queue);

  return solver_ctx;
}

void solver_run(solver_context_t* ctx) {
  OPENCL_CATCH_ERROR(clEnqueueAcquireGLObjects(ctx->queue, 1, &ctx->buffers.Q_next, 0, NULL, NULL));

  size_t origin[] = {0, 0, 0};
  size_t region[] = {WIDTH, HEIGHT, DEPTH};
  OPENCL_CATCH_ERROR(clEnqueueCopyImage(ctx->queue,
					ctx->buffers.Q_next,
					ctx->buffers.Q_prev,
					origin,
					origin,
					region,
					0,
					NULL,
					NULL));

  solver_compute_weno_reconstruction_and_fluxes(ctx, ctx->buffers.Q_prev);

  float S_x_max = solver_get_max_value_image2d(ctx, ctx->buffers.S_x);
  float S_y_max = solver_get_max_value_image2d(ctx, ctx->buffers.S_y);

  ctx->delta_t = ETA*fminf(DELTA_X/S_x_max, DELTA_Y/S_y_max);

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 0, sizeof(cl_mem), &ctx->buffers.Q_1));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 1, sizeof(cl_mem), &ctx->buffers.Q_prev));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 2, sizeof(cl_mem), &ctx->buffers.F));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 3, sizeof(cl_mem), &ctx->buffers.G));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 4, sizeof(float), &ctx->delta_x));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 5, sizeof(float), &ctx->delta_y));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_1, 6, sizeof(float), &ctx->delta_t));

  const size_t global_work_offset[] = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 0};
  const size_t global_work_size[]   = {WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y, 4};
  const size_t local_work_size[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 1};
  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.set_Q_1,
					    3,
					    global_work_offset,
					    global_work_size,
					    local_work_size,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_1);

  solver_compute_weno_reconstruction_and_fluxes(ctx, ctx->buffers.Q_1);

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 0, sizeof(cl_mem), &ctx->buffers.Q_2));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 1, sizeof(cl_mem), &ctx->buffers.Q_prev));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 2, sizeof(cl_mem), &ctx->buffers.Q_1));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 3, sizeof(cl_mem), &ctx->buffers.F));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 4, sizeof(cl_mem), &ctx->buffers.G));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 5, sizeof(float), &ctx->delta_x));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 6, sizeof(float), &ctx->delta_y));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_2, 7, sizeof(float), &ctx->delta_t));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.set_Q_2,
					    3,
					    global_work_offset,
					    global_work_size,
					    local_work_size,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_2);

  solver_compute_weno_reconstruction_and_fluxes(ctx, ctx->buffers.Q_2);

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 0, sizeof(cl_mem), &ctx->buffers.Q_next));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 1, sizeof(cl_mem), &ctx->buffers.Q_prev));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 2, sizeof(cl_mem), &ctx->buffers.Q_1));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 3, sizeof(cl_mem), &ctx->buffers.Q_2));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 4, sizeof(cl_mem), &ctx->buffers.F));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 5, sizeof(cl_mem), &ctx->buffers.G));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 6, sizeof(float), &ctx->delta_x));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 7, sizeof(float), &ctx->delta_y));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_Q_next, 8, sizeof(float), &ctx->delta_t));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.set_Q_next,
					    3,
					    global_work_offset,
					    global_work_size,
					    local_work_size,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_next);

  ctx->t += ctx->delta_t;

  printf("\rt:%f, delta_t:%f", ctx->t, ctx->delta_t);
  fflush(stdout);

  solver_compute_density_range(ctx);

  OPENCL_CATCH_ERROR(clEnqueueReleaseGLObjects(ctx->queue, 1, &ctx->buffers.Q_next, 0, NULL, NULL));
  clFinish(ctx->queue);
}

void solver_deinit(solver_context_t* ctx) {
  printf("\n");

  clReleaseMemObject(ctx->buffers.Q_next);
  clReleaseMemObject(ctx->buffers.Q_prev);
  clReleaseMemObject(ctx->buffers.Q_1);
  clReleaseMemObject(ctx->buffers.Q_2);
  clReleaseMemObject(ctx->buffers.Q_l);
  clReleaseMemObject(ctx->buffers.Q_d);
  clReleaseMemObject(ctx->buffers.Q_u);
  clReleaseMemObject(ctx->buffers.Q_r);
  clReleaseMemObject(ctx->buffers.F);
  clReleaseMemObject(ctx->buffers.G);
  clReleaseMemObject(ctx->buffers.S_x);
  clReleaseMemObject(ctx->buffers.S_y);
  clReleaseMemObject(ctx->buffers.min);
  clReleaseMemObject(ctx->buffers.max);

  clReleaseKernel(ctx->kernels.minmax_image3d_slice);
  clReleaseKernel(ctx->kernels.minmax_buffer);
  clReleaseKernel(ctx->kernels.max_image2d);
  clReleaseKernel(ctx->kernels.max_buffer);
  clReleaseKernel(ctx->kernels.set_periodic_boundary_conditions_x);
  clReleaseKernel(ctx->kernels.set_periodic_boundary_conditions_y);
  clReleaseKernel(ctx->kernels.compute_x_flux);
  clReleaseKernel(ctx->kernels.compute_y_flux);
  clReleaseKernel(ctx->kernels.compute_weno_reconstruction);
  clReleaseKernel(ctx->kernels.init_Q);
  clReleaseKernel(ctx->kernels.set_Q_1);
  clReleaseKernel(ctx->kernels.set_Q_2);
  clReleaseKernel(ctx->kernels.set_Q_next);
  clReleaseProgram(ctx->program);
  clReleaseCommandQueue(ctx->queue);
  clReleaseContext(ctx->context);

  free(ctx);
}

static void cl_context_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data) {
  printf("openCL error %s\n", errinfo);
}

static cl_program build_program_from_source(const char* filepath, const char* build_options, cl_context context, cl_device_id device_id) {
  cl_int error_code = CL_SUCCESS;
  const char* source_filename = "./kernels/kernels.cl";
  char* source_string = read_file(source_filename, NULL);

  cl_program program = clCreateProgramWithSource(context,
						 1,
						 (const char**)&source_string,
						 NULL,
						 &error_code);
  OPENCL_CATCH_ERROR(error_code);

  const char options[] = "-cl-std=CL2.0";
  clBuildProgram(program, 1, &device_id, options, NULL, NULL);
  OPENCL_CATCH_ERROR(error_code);

  cl_build_status build_status;
  clGetProgramBuildInfo(program,
			device_id,
			CL_PROGRAM_BUILD_STATUS,
			sizeof(cl_build_status),
			&build_status,
			NULL);
  if(build_status != CL_BUILD_SUCCESS) {
    fprintf(stderr, "Failed to compile %s:\n", source_filename);

    size_t log_size;
    clGetProgramBuildInfo(program,
			  device_id,
			  CL_PROGRAM_BUILD_LOG,
			  0,
			  NULL,
			  &log_size);
    char* build_log = malloc(sizeof(*build_log)*log_size);
    clGetProgramBuildInfo(program,
			  device_id,
			  CL_PROGRAM_BUILD_LOG,
			  log_size,
			  build_log,
			  NULL);
    fprintf(stderr, "%s", build_log);
    free(build_log);
    free(source_string);

    exit(EXIT_FAILURE);
  }

  return program;
}

static void solver_compute_density_range(solver_context_t* ctx) {
  cl_int error_code = CL_SUCCESS;

  const size_t global_work_offset_first_reduction[] = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 0};
  const size_t global_work_size_first_reduction[]   = {WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y, 1};
  const size_t local_work_size_first_reduction[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 1};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_image3d_slice, 0, sizeof(cl_mem), &ctx->buffers.Q_next));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_image3d_slice, 1, sizeof(float)*LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y, NULL));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_image3d_slice, 2, sizeof(float)*LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y, NULL));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_image3d_slice, 3, sizeof(cl_mem), &ctx->buffers.min));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_image3d_slice, 4, sizeof(cl_mem), &ctx->buffers.max));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.minmax_image3d_slice,
					    3,
					    global_work_offset_first_reduction,
					    global_work_size_first_reduction,
					    local_work_size_first_reduction,
					    0,
					    NULL,
					    NULL));

  const size_t global_work_offset_second_reduction[] = {0};
  const size_t global_work_size_second_reduction[]   = {(WIDTH/LOCAL_WORK_SIZE_X-2)*(HEIGHT/LOCAL_WORK_SIZE_Y-2)};
  const size_t local_work_size_second_reduction[]   = {LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_buffer, 0, sizeof(cl_mem), &ctx->buffers.min));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_buffer, 1, sizeof(cl_mem), &ctx->buffers.max));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_buffer, 2, sizeof(float)*LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y, NULL));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_buffer, 3, sizeof(float)*LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y, NULL));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_buffer, 4, sizeof(cl_mem), &ctx->buffers.min));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.minmax_buffer, 5, sizeof(cl_mem), &ctx->buffers.max));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.minmax_buffer,
					    1,
					    global_work_offset_second_reduction,
					    global_work_size_second_reduction,
					    local_work_size_second_reduction,
					    0,
					    NULL,
					    NULL));

  float* max_values = clEnqueueMapBuffer(ctx->queue, ctx->buffers.max, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*global_work_size_second_reduction[0]/local_work_size_second_reduction[0], 0, NULL, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);

  float max = max_values[0];
  for(size_t i=1; i<global_work_size_second_reduction[0]/local_work_size_second_reduction[0]; i++)
    max = fmaxf(max, max_values[i]);
  ctx->rho_max = max;
  OPENCL_CATCH_ERROR(clEnqueueUnmapMemObject(ctx->queue, ctx->buffers.max, max_values, 0, 0, NULL));

  float* min_values = clEnqueueMapBuffer(ctx->queue, ctx->buffers.min, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*global_work_size_second_reduction[0]/local_work_size_second_reduction[0], 0, NULL, NULL, &error_code);
  OPENCL_CATCH_ERROR(error_code);

  float min = min_values[0];
  for(size_t i=1; i<global_work_size_second_reduction[0]/local_work_size_second_reduction[0]; i++)
    max = fminf(min, min_values[i]);
  ctx->rho_min = min;
  OPENCL_CATCH_ERROR(clEnqueueUnmapMemObject(ctx->queue, ctx->buffers.min, max_values, 0, 0, NULL));
}

float solver_get_min_density(solver_context_t* ctx) {
  return ctx->rho_min;
}
float solver_get_max_density(solver_context_t* ctx) {
  return ctx->rho_max;
}

static void solver_set_periodic_boundary_conditions(solver_context_t* ctx, cl_mem image) {

  const size_t global_work_offset_x[] = {0, 0};
  const size_t global_work_size_x[]   = {LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y};
  const size_t local_work_size_x[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_periodic_boundary_conditions_x, 0, sizeof(cl_mem), &image));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.set_periodic_boundary_conditions_x,
					    2,
					    global_work_offset_x,
					    global_work_size_x,
					    local_work_size_x,
					    0,
					    NULL,
					    NULL));

  const size_t global_work_offset_y[] = {0, 0};
  const size_t global_work_size_y[]   = {WIDTH, LOCAL_WORK_SIZE_Y};
  const size_t local_work_size_y[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.set_periodic_boundary_conditions_y, 0, sizeof(cl_mem), &image));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.set_periodic_boundary_conditions_y,
					    2,
					    global_work_offset_y,
					    global_work_size_y,
					    local_work_size_y,
					    0,
					    NULL,
					    NULL));
  clFinish(ctx->queue);
}

static float solver_get_max_value_image2d(solver_context_t* ctx, cl_mem image) {
  cl_int error_code = CL_SUCCESS;

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.max_image2d, 0, sizeof(cl_mem), &image));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.max_image2d, 1, sizeof(float)*LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y, NULL));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.max_image2d, 2, sizeof(cl_mem), &ctx->buffers.max));

  const size_t global_work_offset_first_reduction[] = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 0};
  const size_t global_work_size_first_reduction[]   = {WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y, 1};
  const size_t local_work_size_first_reduction[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 1};

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.max_image2d,
					    2,
					    global_work_offset_first_reduction,
					    global_work_size_first_reduction,
					    local_work_size_first_reduction,
					    0,
					    NULL,
					    NULL));

  const size_t global_work_offset_second_reduction[] = {0};
  const size_t global_work_size_second_reduction[]   = {(WIDTH/LOCAL_WORK_SIZE_X-2)*(HEIGHT/LOCAL_WORK_SIZE_Y-2)};
  const size_t local_work_size_second_reduction[]   = {LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.max_buffer, 0, sizeof(cl_mem), &ctx->buffers.max));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.max_buffer, 1, sizeof(float)*LOCAL_WORK_SIZE_X*LOCAL_WORK_SIZE_Y, NULL));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.max_buffer, 2, sizeof(cl_mem), &ctx->buffers.max));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.max_buffer,
					    1,
					    global_work_offset_second_reduction,
					    global_work_size_second_reduction,
					    local_work_size_second_reduction,
					    0,
					    NULL,
					    NULL));

  float* max_values = clEnqueueMapBuffer(ctx->queue, ctx->buffers.max, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*global_work_size_second_reduction[0]/local_work_size_second_reduction[0], 0, NULL, NULL, &error_code);
  float max = max_values[0];
  for(size_t i=1; i<global_work_size_second_reduction[0]/local_work_size_second_reduction[0]; i++)
    max = fmaxf(max, max_values[i]);

  OPENCL_CATCH_ERROR(clEnqueueUnmapMemObject(ctx->queue, ctx->buffers.max, max_values, 0, 0, NULL));
  return max;
}

static void solver_compute_weno_reconstruction_and_fluxes(solver_context_t* ctx, cl_mem Q) {

  const size_t global_work_offset_weno[] = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 0};
  const size_t global_work_size_weno[]   = {WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y, 4};
  const size_t local_work_size_weno[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y, 1};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_weno_reconstruction, 0, sizeof(cl_mem), &Q));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_weno_reconstruction, 1, sizeof(cl_mem), &ctx->buffers.Q_l));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_weno_reconstruction, 2, sizeof(cl_mem), &ctx->buffers.Q_r));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_weno_reconstruction, 3, sizeof(cl_mem), &ctx->buffers.Q_d));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_weno_reconstruction, 4, sizeof(cl_mem), &ctx->buffers.Q_u));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.compute_weno_reconstruction,
					    3,
					    global_work_offset_weno,
					    global_work_size_weno,
					    local_work_size_weno,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_l);
  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_r);
  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_d);
  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.Q_u);

  const size_t global_work_offset[] = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y};
  const size_t global_work_size[]   = {WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y};
  const size_t local_work_size[]    = {LOCAL_WORK_SIZE_X, LOCAL_WORK_SIZE_Y};

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_x_flux, 0, sizeof(cl_mem), &ctx->buffers.Q_l));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_x_flux, 1, sizeof(cl_mem), &ctx->buffers.Q_r));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_x_flux, 2, sizeof(cl_mem), &ctx->buffers.F));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_x_flux, 3, sizeof(cl_mem), &ctx->buffers.S_x));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.compute_x_flux,
					    2,
					    global_work_offset,
					    global_work_size,
					    local_work_size,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.F);

  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_y_flux, 0, sizeof(cl_mem), &ctx->buffers.Q_d));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_y_flux, 1, sizeof(cl_mem), &ctx->buffers.Q_u));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_y_flux, 2, sizeof(cl_mem), &ctx->buffers.G));
  OPENCL_CATCH_ERROR(clSetKernelArg(ctx->kernels.compute_y_flux, 3, sizeof(cl_mem), &ctx->buffers.S_y));

  OPENCL_CATCH_ERROR(clEnqueueNDRangeKernel(ctx->queue,
					    ctx->kernels.compute_y_flux,
					    2,
					    global_work_offset,
					    global_work_size,
					    local_work_size,
					    0,
					    NULL,
					    NULL));

  solver_set_periodic_boundary_conditions(ctx, ctx->buffers.G);
}
