#ifndef PREVIEWER_H
#define PREVIEWER_H

#include <SDL2/SDL.h>
#include <GL/gl.h>

/** opaque datatype representing the state of the renderer */
typedef struct preview_context_t preview_context_t;
typedef struct solver_context_t solver_context_t;

/** Initialize openGL context and create a 3d texture that will be
 * shared with openCL */
preview_context_t* previewer_init(void);

/** renders the shared texture to the screen */
void previewer_render_frame(preview_context_t* context, solver_context_t* solver_context);

/** returns the current openGL context */
SDL_GLContext* previewer_get_gl_context(preview_context_t* context);

/** returns the X11 window handle */
void* previewer_get_display(preview_context_t* context);

/** returns the rendered texture */
GLuint previewer_get_texture(preview_context_t* context);

/** Cleanup openGL context and associated objects */
void previewer_deinit(preview_context_t* context);

#endif // PREVIEWER_H
