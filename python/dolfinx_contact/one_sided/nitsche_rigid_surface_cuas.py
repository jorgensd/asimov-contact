# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple, Dict

import basix
import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.log as _log
import dolfinx.mesh as _mesh
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import ufl
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)

__all__ = ["nitsche_rigid_surface_cuas"]
kt = dolfinx_contact.cpp.Kernel


def nitsche_rigid_surface_cuas(mesh: _mesh.Mesh, mesh_data: Tuple[_mesh.MeshTags, int, int, int, int],
                               physical_parameters: dict = {}, nitsche_parameters: Dict[str, float] = {},
                               vertical_displacement: float = -0.1, nitsche_bc: bool = True, quadrature_degree: int = 5,
                               form_compiler_parameters: Dict = {}, jit_parameters: Dict = {}, petsc_options: Dict = {},
                               newton_options: Dict = {}):
    """
    Use custom kernel to compute the one sided contact problem with a mesh coming into contact
    with a rigid surface (meshed).

    Parameters
    ==========
    mesh
        The input mesh
    mesh_data
        A quinteplet with a mesh tag for facets and values v0, v1, v2, v3. v0 and v3
        should be the values in the mesh tags for facets to apply a Dirichlet condition
        on, where v0 corresponds to the elastic body and v2 to the rigid body. v1 is the
        value for facets which should have applied a contact condition on and v2 marks
        the potential contact surface on the rigid body.
    physical_parameters
        Optional dictionary with information about the linear elasticity problem.
        Valid (key, value) tuples are: ('E': float), ('nu', float), ('strain', bool)
    nitsche_parameters
        Optional dictionary with information about the Nitsche configuration.
        Valid (keu, value) tuples are: ('gamma', float), ('theta', float) where theta can be -1, 0 or 1 for
        skew-symmetric, penalty like or symmetric enforcement of Nitsche conditions
    vertical_displacement
        The amount of verticial displacment enforced on Dirichlet boundary
    nitsche_bc
        Use Nitche's method to enforce Dirichlet boundary conditions
    quadrature_degree
        The quadrature degree to use for the custom contact kernels
    form_compiler_parameters
        Parameters used in FFCX compilation of this form. Run `ffcx --help` at
        the commandline to see all available opsurface_bottomtions. Takes priority over all
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

    # Nitsche parameters and variables
    theta = nitsche_parameters.get("theta", 1)
    gamma = nitsche_parameters.get("gamma", 10)

    # Unpack mesh data
    (facet_marker, dirichlet_value_elastic, contact_value_elastic, contact_value_rigid,
     dirichlet_value_rigid) = mesh_data
    assert(facet_marker.dim == mesh.topology.dim - 1)
    gdim = mesh.geometry.dim

    # Setup function space and functions used in Jacobian and residual formulation
    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    u = _fem.Function(V)
    v = ufl.TestFunction(V)

    # Compute classical (volume) contributions of the equations of linear elasticity
    dx = ufl.Measure("dx", domain=mesh)
    J = ufl.inner(sigma(du), epsilon(v)) * dx
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
        h = ufl.Circumradius(mesh)
        n = ufl.FacetNormal(mesh)

        disp_vec = np.zeros(gdim)
        disp_vec[gdim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        F += - ufl.inner(sigma(u) * n, v) * ds(dirichlet_value_elastic)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(dirichlet_value_elastic) + E * gamma / h * ufl.inner(u - u_D, v) * ds(dirichlet_value_elastic)

        J += - ufl.inner(sigma(du) * n, v) * ds(dirichlet_value_elastic)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(dirichlet_value_elastic) + E * gamma / h * ufl.inner(du, v) * ds(dirichlet_value_elastic)

        # Nitsche bc for rigid plane
        disp_plane = np.zeros(gdim)
        u_D_plane = ufl.as_vector(disp_plane)
        F += - ufl.inner(sigma(u) * n, v) * ds(dirichlet_value_rigid)\
             - theta * ufl.inner(sigma(v) * n, u - u_D_plane) * \
            ds(dirichlet_value_rigid) + E * gamma / h * ufl.inner(u - u_D_plane, v) * ds(dirichlet_value_rigid)
        J += - ufl.inner(sigma(du) * n, v) * ds(dirichlet_value_rigid)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(dirichlet_value_rigid) + E * gamma / h * ufl.inner(du, v) * ds(dirichlet_value_rigid)
    else:
        print("Dirichlet bc not implemented in custom assemblers yet.")

    # Custom assembly of contact boundary conditions
    _log.set_log_level(_log.LogLevel.OFF)  # avoid large amounts of output
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, quadrature_degree,
                                             mesh.topology.dim - 1, basix.QuadratureType.Default)
    consts = np.array([gamma * E, theta])

    # Compute coefficients for mu and lambda as DG-0 functions
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu2 = _fem.Function(V2)
    mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))

    # Compute integral entities on exterior facets (cell_index, local_index)
    contact_facets = facet_marker.indices[facet_marker.values == contact_value_elastic]
    integral = _fem.IntegralType.exterior_facet
    integral_entities = dolfinx_cuas.cpp.compute_active_entities(mesh, contact_facets, integral)

    # Pack mu and lambda on facets
    coeffs = dolfinx_cuas.pack_coefficients([mu2, lmbda2], integral_entities)
    # Pack circumradius on facets
    h_facets = dolfinx_contact.pack_circumradius_facet(mesh, contact_facets)

    # Create contact class
    contact = dolfinx_contact.cpp.Contact(facet_marker, [contact_value_elastic, contact_value_rigid], V._cpp_object)
    contact.set_quadrature_degree(quadrature_degree)

    # Compute gap and normals
    contact.create_distance_map(0, 1)
    g_vec = contact.pack_gap(0)
    n_surf = contact.pack_ny(0, g_vec)

    # Concatenate coefficients
    coeffs = np.hstack([coeffs, h_facets, g_vec, n_surf])

    # Create RHS kernels
    F_cuas = _fem.form(F, jit_parameters=jit_parameters, form_compiler_parameters=form_compiler_parameters)
    kernel_rhs = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.Rhs, q_rule,
                                                             [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object],
                                                             False)

    def create_b():
        return _fem.create_vector(F_cuas)

    def F(x, b):
        u.vector[:] = x.array
        u_packed = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object], integral_entities)
        c = np.hstack([u_packed, coeffs])
        dolfinx_cuas.assemble_vector(b, V, contact_facets, kernel_rhs, c, consts, integral)
        _fem.assemble_vector(b, F_cuas)

    # Create Jacobian kernels
    J_cuas = _fem.form(J, jit_parameters=jit_parameters, form_compiler_parameters=form_compiler_parameters)
    kernel_J = dolfinx_contact.cpp.generate_contact_kernel(
        V._cpp_object, kt.Jac, q_rule, [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object], False)

    def create_A():
        return _fem.create_matrix(J_cuas)

    def A(x, A):
        u.vector[:] = x.array
        u_packed = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object], integral_entities)
        c = np.hstack([u_packed, coeffs])
        dolfinx_cuas.assemble_matrix(A, V, contact_facets, kernel_J, c, consts, integral)
        _fem.assemble_matrix(A, J_cuas)

    # Setup non-linear problem and Newton-solver
    problem = dolfinx_cuas.NonlinearProblemCUAS(F, A, create_b, create_A)
    solver = dolfinx_cuas.NewtonSolver(mesh.comm, problem)

    # Create rigid motion null-space
    null_space = rigid_motions_nullspace(V)
    solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    solver.atol = newton_options.get("atol", 1e-9)
    solver.rtol = newton_options.get("rtol", 1e-9)
    solver.convergence_criterion = newton_options.get("convergence_criterion", "incremental")
    solver.max_it = newton_options.get("max_it", 50)
    solver.error_on_nonconvergence = newton_options.get("error_on_nonconvergence", True)
    solver.relaxation_parameter = newton_options.get("relaxation_parameter", 1.0)

    # Set initial condition
    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        values[-1] = -vertical_displacement
        return values
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

    dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    _log.set_log_level(_log.LogLevel.INFO)

    # Solve non-linear problem
    with _common.Timer(f"{dofs_global} Solve Nitsche"):
        n, converged = solver.solve(u)
    u.x.scatter_forward()

    if solver.error_on_nonconvergence:
        assert(converged)
    print(f"{dofs_global}, Number of interations: {n:d}")

    return u
