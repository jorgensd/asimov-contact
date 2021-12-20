# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple, Dict

import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.log as _log
import dolfinx.mesh as dmesh
import dolfinx.nls as _nls
import numpy as np
import ufl
from petsc4py import PETSc as _PETSc

from dolfinx_contact.helpers import (R_minus, epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)

__all__ = ["nitsche_ufl"]


def nitsche_ufl(mesh: dmesh.Mesh, mesh_data: Tuple[dmesh.MeshTags, int, int],
                physical_parameters: dict = {}, nitsche_parameters: Dict[str, float] = {},
                plane_loc: float = 0.0, vertical_displacement: float = -0.1,
                nitsche_bc: bool = True, quadrature_degree: int = 5, form_compiler_parameters: Dict = {},
                jit_parameters: Dict = {}, petsc_options: Dict = {}, newton_options: Dict = {}) -> _fem.Function:
    """
    Use UFL to compute the one sided contact problem with a mesh coming into contact
    with a rigid surface (not meshed).

    Parameters
    ==========
    mesh
        The input mesh
    mesh_data
        A triplet with a mesh tag for facets and values v0, v1. v0 should be the value in the mesh tags
        for facets to apply a Dirichlet condition on. v1 is the value for facets which should have applied
        a contact condition on
    physical_parameters
        Optional dictionary with information about the linear elasticity problem.
        Valid (key, value) tuples are: ('E': float), ('nu', float), ('strain', bool)
    nitsche_parameters
        Optional dictionary with information about the Nitsche configuration.
        Valid (keu, value) tuples are: ('gamma', float), ('theta', float) where theta can be -1, 0 or 1 for
        skew-symmetric, penalty like or symmetric enforcement of Nitsche conditions
    plane_loc
        The location of the plane in y-coordinate (2D) and z-coordinate (3D)
    vertical_displacement
        The amount of verticial displacment enforced on Dirichlet boundary
    nitsche_bc
        Use Nitche's method to enforce Dirichlet boundary conditions
    quadrature_degree
        The quadrature degree to use for the custom contact kernels
    form_compiler_parameters
        Parameters used in FFCX compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINX.
    jit_parameters
        Parameters used in CFFI JIT compilation of C code generated by FFCX.
        See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py
        for all available parameters. Takes priority over all other parameter values.
    petsc_options
        Parameters that is passed to the linear algebra backend
        PETSc. For available choices for the 'petsc_options' kwarg,
        see the `PETSc-documentation
        <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`
    newton_options
        Dictionary with Newton-solver options. Valid (key, item) tuples are:
        ("atol", float), ("rtol", float), ("convergence_criterion", "str"),
        ("max_it", int), ("error_on_nonconvergence", bool), ("relaxation_parameter", float)
    """
    # Compute lame parameters
    plane_strain = physical_parameters.get("strain", False)
    E = physical_parameters.get("E", 1e3)
    nu = physical_parameters.get("nu", 0.1)
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Nitche parameters and variables
    theta = nitsche_parameters.get("theta", 1)
    gamma = nitsche_parameters.get("gamma", 1)

    (facet_marker, top_value, bottom_value) = mesh_data
    assert(facet_marker.dim == mesh.topology.dim - 1)

    # Normal vector pointing into plane (but outward of the body coming into contact)
    # Similar to computing the normal by finding the gap vector between two meshes
    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = -1
    n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)

    # Scaled Nitsche parameter
    h = ufl.Circumradius(mesh)
    gamma_scaled = gamma * E / h

    # Mimicking the plane y=-plane_loc
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + plane_loc
    g_vec = [i for i in range(mesh.geometry.dim)]
    g_vec[mesh.geometry.dim - 1] = gap

    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = _fem.Function(V)
    v = ufl.TestFunction(V)

    metadata = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(_fem.Constant(mesh, np.float64([0, ] * mesh.geometry.dim)), v) * dx

    # Derivation of one sided Nitsche with gap function
    n = ufl.FacetNormal(mesh)

    def sigma_n(v):
        # NOTE: Different normals, see summary paper
        return ufl.dot(sigma(v) * n, n_2)
    F = a - theta / gamma_scaled * sigma_n(u) * sigma_n(v) * ds(bottom_value) - L
    F += 1 / gamma_scaled * R_minus(sigma_n(u) + gamma_scaled * (gap - ufl.dot(u, n_2))) * \
        (theta * sigma_n(v) - gamma_scaled * ufl.dot(v, n_2)) * ds(bottom_value)

    # Compute corresponding Jacobian
    du = ufl.TrialFunction(V)
    q = sigma_n(u) + gamma_scaled * (gap - ufl.dot(u, n_2))
    J = ufl.inner(sigma(du), epsilon(v)) * ufl.dx - theta / gamma_scaled * sigma_n(du) * sigma_n(v) * ds(bottom_value)
    J += 1 / gamma_scaled * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) - gamma_scaled * ufl.dot(du, n_2)) * \
        (theta * sigma_n(v) - gamma_scaled * ufl.dot(v, n_2)) * ds(bottom_value)

    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(mesh.geometry.dim)
        disp_vec[mesh.geometry.dim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        F += - ufl.inner(sigma(u) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(top_value) + gamma_scaled / h * ufl.inner(u - u_D, v) * ds(top_value)
        bcs = []
        J += - ufl.inner(sigma(du) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(top_value) + gamma_scaled / h * ufl.inner(du, v) * ds(top_value)
    else:
        # strong Dirichlet boundary conditions
        def _u_D(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[mesh.geometry.dim - 1] = vertical_displacement
            return values
        u_D = _fem.Function(V)
        u_D.interpolate(_u_D)
        u_D.name = "u_D"
        u_D.x.scatter_forward()
        tdim = mesh.topology.dim
        dirichlet_dofs = _fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
        bc = _fem.DirichletBC(u_D, dirichlet_dofs)
        bcs = [bc]

    # DEBUG: Write each step of Newton iterations
    # Create nonlinear problem and Newton solver
    # def form(self, x: _PETSc.Vec):
    #     x.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
    #     self.i += 1
    #     xdmf.write_function(u, self.i)

    # setattr(_fem.NonlinearProblem, "form", form)

    problem = _fem.NonlinearProblem(F, u, bcs, J=J, jit_parameters=jit_parameters,
                                    form_compiler_parameters=form_compiler_parameters)

    # DEBUG: Write each step of Newton iterations
    # problem.i = 0
    # xdmf = _io.XDMFFile(mesh.comm, "results/tmp_sol.xdmf", "w")
    # xdmf.write_mesh(mesh)

    solver = _nls.NewtonSolver(mesh.comm, problem)
    null_space = rigid_motions_nullspace(V)
    solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    solver.atol = newton_options.get("atol", 1e-9)
    solver.rtol = newton_options.get("rtol", 1e-9)
    solver.convergence_criterion = newton_options.get("convergence_criterion", "incremental")
    solver.max_it = newton_options.get("max_it", 50)
    solver.error_on_nonconvergence = newton_options.get("error_on_nonconvergence", True)
    solver.relaxation_parameter = newton_options.get("relaxation_parameter", 0.8)

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - plane_loc
        return values

    # Set initial_condition:
    u.interpolate(_u_initial)

    # Define solver and options
    ksp = solver.krylov_solver
    opts = _PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    # Set PETSc options
    opts = _PETSc.Options()
    opts.prefixPush(option_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    # Solve non-linear problem
    _log.set_log_level(_log.LogLevel.INFO)
    num_dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    with _common.Timer(f"{num_dofs_global} Solve Nitsche"):
        n, converged = solver.solve(u)
    u.x.scatter_forward()
    if solver.error_on_nonconvergence:
        assert(converged)
    print(f"{num_dofs_global}, Number of interations: {n:d}")
    return u
