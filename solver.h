#ifndef SOLVER_H
#define SOLVER_H

#include "previewer.h"

typedef struct solver_context_t solver_context_t;

/** Initialize openCL context using the openCL icd-loader provided in
 *libOpenGL.so with openGL interoperability */
solver_context_t* solver_init(preview_context_t* preview_ctx);

/** Advance a step in the simulation */
void solver_run(solver_context_t* context);

/** returns the min and max values of the density of the previous
 * timestep */
float solver_get_min_density(solver_context_t* context);
float solver_get_max_density(solver_context_t* context);

/** Cleanup openCL context and associated objects */
void solver_deinit(solver_context_t* context);

#endif // SOLVER_H
