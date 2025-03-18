cmake_minimum_required(VERSION 3.22)

project(openCL-FV-solver
  DESCRIPTION "A compressible Euler equation GPU finite volume solver to simulate the Kelvin-Helmholtz instability"
  LANGUAGES C)

add_executable(program src/main.c src/utilities.c src/solver.c src/previewer.c)

set_property(TARGET program PROPERTY C_STANDARD 11)
target_compile_definitions(program PUBLIC CL_TARGET_OPENCL_VERSION=220)
target_link_libraries(program OpenCL GL SDL2 GLEW EGL m)
