#include <SDL2/SDL_syswm.h>

#include <GL/glew.h>

#include <stdio.h>
#include <stdbool.h>

#include "global.h"
#include "previewer.h"
#include "utilities.h"
#include "solver.h"

typedef struct preview_context_t {
  struct sdl_context_t {
    SDL_Window* window;
  } sdl;
  struct {
    SDL_GLContext* context;
    GLuint vertex_position_buffer;
    GLuint vertex_texture_coordinate_buffer;
    GLuint vertex_array_object;
    GLuint vertex_shader;
    GLuint fragment_shader;
    GLuint program;
    GLuint texture;
    GLuint sampler;
    struct {
      GLint min_location;
      GLint max_location;
    } uniforms;
  } gl;
} preview_context_t;

#ifdef DEBUG
/** debug callback function called if GL_DEBUG_OUTPUT is set */
static void opengl_debug_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity,
					  GLsizei length, const GLchar* message, const void* userParam);
#endif

/** returns an GLuint representing the compiled shader source if
 * sucessful. Otherwise the program exits with EXIT_FAILURE */
static GLuint compile_shader(const char* shader_source, GLenum shader_type);

/** returns an GLuint representing the linked program of `nb_shaders'
 * compiled shaders if sucessful. Otherwise the program exits with
 * EXIT_FAILURE */
static GLuint link_program(GLuint* shaders, size_t nb_shaders);

preview_context_t* previewer_init(void) {
  preview_context_t* ctx = malloc(sizeof(*ctx));

  if(!SDL_SetHint(SDL_HINT_VIDEO_X11_FORCE_EGL, "SDL_TRUE")) {
    printf("tsetsetsttst\n");
    exit(EXIT_FAILURE);
   }

 ctx->sdl.window = SDL_CreateWindow("Kelvin-Helmholtz instability",
				    SDL_WINDOWPOS_CENTERED,
				    SDL_WINDOWPOS_CENTERED,
				    SCREEN_WIDTH,
				    SCREEN_HEIGHT,
				    SDL_WINDOW_OPENGL);

  ctx->gl.context = SDL_GL_CreateContext(ctx->sdl.window);
  SDL_GL_SetSwapInterval(0);

  if(glewInit()){
    fprintf(stderr, "Unable to initialize GLEW ... exiting\n");
    exit(EXIT_FAILURE);
  }

  glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef DEBUG
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(opengl_debug_message_callback, 0);
#endif

  glGenVertexArrays(1, &ctx->gl.vertex_array_object);
  glBindVertexArray(ctx->gl.vertex_array_object);

  const size_t NUM_BUFFERS = 2;
  GLuint vertex_buffers[NUM_BUFFERS];
  glGenBuffers(NUM_BUFFERS, vertex_buffers);
  
  ctx->gl.vertex_position_buffer = vertex_buffers[0];
  ctx->gl.vertex_texture_coordinate_buffer = vertex_buffers[1];

  const GLfloat vertex_position[] = {-1.0f, -1.0f,
				     -1.0f,  1.0f,
				      1.0,  -1.0,
				      1.0,   1.0};

  GLfloat vertex_texture_coordinate[] = {0.0f, 0.0f,
					 1.0f, 0.0f,
					 0.0f, 1.0f,
					 1.0f, 1.0f};

  glBindBuffer(GL_ARRAY_BUFFER, ctx->gl.vertex_position_buffer);
  glBufferData(GL_ARRAY_BUFFER,
	       sizeof(vertex_position),
	       vertex_position,
	       GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, ctx->gl.vertex_texture_coordinate_buffer);
  glBufferData(GL_ARRAY_BUFFER,
	       sizeof(vertex_texture_coordinate),
	       vertex_texture_coordinate,
	       GL_STATIC_DRAW);

  glGenTextures(1, &ctx->gl.texture);
  glBindTexture(GL_TEXTURE_2D, ctx->gl.texture);
  glActiveTexture(GL_TEXTURE0);

  glGenSamplers(1, &ctx->gl.sampler);
  glBindSampler(0, ctx->gl.sampler);
  glSamplerParameteri(ctx->gl.sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glSamplerParameteri(ctx->gl.sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, WIDTH-2*LOCAL_WORK_SIZE_X, HEIGHT-2*LOCAL_WORK_SIZE_Y);

  char* vertex_shader_source = read_file("./kernels/vertex.glsl", NULL);
  char* fragment_shader_source = read_file("./kernels/fragment.glsl", NULL);

  ctx->gl.vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER);
  ctx->gl.fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER);

  free(vertex_shader_source);
  free(fragment_shader_source);

  GLuint shaders[2] = {ctx->gl.vertex_shader, ctx->gl.fragment_shader};
  ctx->gl.program = link_program(shaders, 2);

  glUseProgram(ctx->gl.program);

  GLint sampler_location = glGetUniformLocation(ctx->gl.program, "tex");
  glUniform1i(sampler_location, 0);

  ctx->gl.uniforms.min_location = glGetUniformLocation(ctx->gl.program, "min");
  ctx->gl.uniforms.max_location = glGetUniformLocation(ctx->gl.program, "max");

  GLint position_location = glGetAttribLocation(ctx->gl.program, "vPosition");
  GLint texture_coordinate_location = glGetAttribLocation(ctx->gl.program, "vTextureCoord");

  glBindBuffer(GL_ARRAY_BUFFER, ctx->gl.vertex_position_buffer);
  glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (GLvoid*) 0);
  glEnableVertexAttribArray(position_location);

  glBindBuffer(GL_ARRAY_BUFFER, ctx->gl.vertex_texture_coordinate_buffer);
  glVertexAttribPointer(texture_coordinate_location, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (GLvoid*) 0);
  glEnableVertexAttribArray(texture_coordinate_location);

  return ctx;
}

void previewer_render_frame(preview_context_t* preview_ctx, solver_context_t* solver_ctx) {
  float min_density = solver_get_min_density(solver_ctx);
  float max_density = solver_get_max_density(solver_ctx);

  glUseProgram(preview_ctx->gl.program);
  glClear(GL_COLOR_BUFFER_BIT);
  glBindVertexArray(preview_ctx->gl.vertex_array_object);
  glUniform1f(preview_ctx->gl.uniforms.min_location, min_density);
  glUniform1f(preview_ctx->gl.uniforms.max_location, max_density);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glFinish();
  SDL_GL_SwapWindow(preview_ctx->sdl.window);
}

SDL_GLContext* previewer_get_gl_context(preview_context_t* context) {
  return context->gl.context;
}

void* previewer_get_display(preview_context_t* context) {
  SDL_SysWMinfo info;
  SDL_VERSION(&info.version);
  if(!(SDL_GetWindowWMInfo(context->sdl.window, &info))){
    const char* error_message = SDL_GetError();
        fprintf(stderr, "Could not resolve WM info struct from SDL2 context: %s\n", error_message);
      exit(EXIT_FAILURE);
    }
  if(info.subsystem == SDL_SYSWM_WAYLAND){
    printf("using wayland\n");
    return info.info.wl.display;
  }else if(info.subsystem == SDL_SYSWM_X11){
    return info.info.x11.display;
  }else{
    fprintf(stderr, "Could not determine windowing system\n");
    exit(EXIT_FAILURE);
  }
}

GLuint previewer_get_texture(preview_context_t* context) {
  return context->gl.texture;
}

void previewer_deinit(preview_context_t* ctx) {
  glDeleteProgram(ctx->gl.program);
  glDeleteShader(ctx->gl.vertex_shader);
  glDeleteShader(ctx->gl.fragment_shader);

  glDeleteVertexArrays(1, &ctx->gl.vertex_array_object);
  glDeleteBuffers(1, &ctx->gl.vertex_position_buffer);

  glDeleteSamplers(1, &ctx->gl.sampler);
  glDeleteTextures(1, &ctx->gl.texture);

  SDL_DestroyWindow(ctx->sdl.window);

  free(ctx);
}

static GLuint compile_shader(const char* shader_source, GLenum shader_type) {
  GLuint shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &shader_source, NULL);
  glCompileShader(shader);
  
  GLint compile_status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
  if(compile_status != GL_TRUE){
    GLint length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    GLchar* buffer = malloc(length*sizeof(*buffer));
    
    glGetShaderInfoLog(shader, length, NULL, buffer);

    fprintf(stderr, "error while compiling shader:\n%s", buffer);
    free(buffer);
    exit(EXIT_FAILURE);
  }
  return shader;
}

static GLuint link_program(GLuint* shaders, size_t nb_shaders) {
  GLuint program = glCreateProgram();

  for(size_t i=0; i<nb_shaders; i++){
    glAttachShader(program, shaders[i]);
  }
  glLinkProgram(program);

  GLint link_status;
  glGetProgramiv(program, GL_LINK_STATUS, &link_status);
  if(link_status != GL_TRUE){
    GLint length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

    GLchar* buffer = malloc(length*sizeof(*buffer));
    glGetProgramInfoLog(program, length, NULL, buffer);

    fprintf(stderr, "error while linking shader:\n%s", buffer);
    free(buffer);
    exit(EXIT_FAILURE);
  }
  return program;
}

#ifdef DEBUG
static void opengl_debug_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
  fprintf(stderr, "OpenGL error %s\n", message);
}
#endif
