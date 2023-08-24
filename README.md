# Compressible Euler equation openCL numerical solver

![Kelvin-Helmholtz instability](./example.png)

A compressible Euler equation GPU finite volume solver to simulate the Kelvin-Helmholtz instability. The numerical scheme is based on a 3rd order TVD Runge-Kutta time stepping with 5th order WENO reconstruction and HLL flux solver inspired by [Evaluation of Riemann flux solvers for WENO reconstruction schemes: Kelvin–Helmholtz instability](https://doi.org/10.1016/j.compfluid.2015.04.026). A brief overview of how we use openCL datatypes is given in `kernels/kernels.cl`.

## Dependencies

- openCL 2.0 or above
- openGL 4.3 or above and a GPU with the `cl_khr_gl_sharing` extension
- GLEW compiled with `-DGLEW_EGL`
- SDL2

## Build

```shell
mkdir build
make all
```

