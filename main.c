#include <stdbool.h>
#include <stdio.h>

#include "previewer.h"
#include "solver.h"

int main(void) {

  preview_context_t* preview_ctx = previewer_init();
  solver_context_t* solver_ctx = solver_init(preview_ctx);

  bool quit = false;
  while(!quit) {
    solver_run(solver_ctx);
    previewer_render_frame(preview_ctx, solver_ctx);

    SDL_Event e;
    while(SDL_PollEvent(&e)){
      if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_q)){
	quit = true;
      }
    }
  }

  solver_deinit(solver_ctx);
  previewer_deinit(preview_ctx);

  return 0;
}

