# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple, Dict

import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.log as _log
import dolfinx.mesh as _mesh
import dolfinx_cuas
import numpy as np
import ufl
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import epsilon, lame_parameters, sigma_func, rigid_motions_nullspace

kt = dolfinx_contact.cpp.Kernel


def nitsche_unbiased(mesh: _mesh.Mesh, mesh_data: Tuple[_mesh.MeshTags, int, int, int, int],
                     physical_parameters: dict = {}, nitsche_parameters: Dict[str, float] = {},
                     vertical_displacement: float = -0.1, nitsche_bc: bool = True, quadrature_degree: int = 5,
                     form_compiler_parameters: Dict = {}, jit_parameters: Dict = {}, petsc_options: Dict = {},
                     newton_options: Dict = {}, initGuess=None):
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    mesh
        The input mesh
    mesh_data
        A quinteplet with a mesh tag for facets and values v0, v1, v2, v3. v0
        and v3 should be the values in the mesh tags for facets to apply a Dirichlet
        condition on, where v0 corresponds to the first elastic body and v2 to the second.
        v1 is the value for facets which should have applied a contact condition on and v2
        marks the potential contact surface on the rigid body.
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
    gamma = E * nitsche_parameters.get("gamma", 10)

    # Unpack mesh data
    (facet_marker, dirichlet_value_0, surface_value_0, surface_value_1, dirichlet_value_1) = mesh_data
    assert(facet_marker.dim == mesh.topology.dim - 1)

    # Functions space and FEM functions
    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    gdim = mesh.geometry.dim
    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    h = ufl.Circumradius(mesh)
    n = ufl.FacetNormal(mesh)
    # Integration measure and ufl part of linear/bilinear form
    metadata = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata,
                     subdomain_data=facet_marker)
    J = ufl.inner(sigma(du), epsilon(v)) * dx - 0.5 * theta * h / gamma * ufl.inner(sigma(du) * n, sigma(v) * n) * \
        ds(surface_value_0) - 0.5 * theta * h / gamma * ufl.inner(sigma(du) * n, sigma(v) * n) * ds(surface_value_1)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
        ds(surface_value_0) - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * ds(surface_value_1)

    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(gdim)
        disp_vec[gdim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        F += - ufl.inner(sigma(u) * n, v) * ds(dirichlet_value_0)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(dirichlet_value_0) + gamma / h * ufl.inner(u - u_D, v) * ds(dirichlet_value_0)

        J += - ufl.inner(sigma(du) * n, v) * ds(dirichlet_value_0)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(dirichlet_value_0) + gamma / h * ufl.inner(du, v) * ds(dirichlet_value_0)
        # Nitsche bc for rigid plane
        disp_plane = np.zeros(gdim)
        # disp_plane[gdim - 1] = -0.5 * vertical_displacement
        u_D_plane = ufl.as_vector(disp_plane)
        F += - ufl.inner(sigma(u) * n, v) * ds(dirichlet_value_1)\
             - theta * ufl.inner(sigma(v) * n, u - u_D_plane) * \
            ds(dirichlet_value_1) + gamma / h * ufl.inner(u - u_D_plane, v) * ds(dirichlet_value_1)
        J += - ufl.inner(sigma(du) * n, v) * ds(dirichlet_value_1)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(dirichlet_value_1) + gamma / h * ufl.inner(du, v) * ds(dirichlet_value_1)
    else:
        raise RuntimeError("Strong Dirichlet bc's are not implemented in custom assemblers yet.")

    # Custom assembly
    _log.set_log_level(_log.LogLevel.OFF)
    # create contact class
    contact = dolfinx_contact.cpp.Contact(facet_marker, [surface_value_0, surface_value_1], V._cpp_object)
    contact.set_quadrature_degree(quadrature_degree)
    contact.create_distance_map(0, 1)
    contact.create_distance_map(1, 0)
    # pack constants
    consts = np.array([gamma, theta])

    # Pack material parameters mu and lambda on each contact surface
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu2 = _fem.Function(V2)
    mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    facets_0 = facet_marker.indices[facet_marker.values == surface_value_0]
    facets_1 = facet_marker.indices[facet_marker.values == surface_value_1]

    integral = _fem.IntegralType.exterior_facet
    entities_0 = dolfinx_cuas.compute_active_entities(mesh, facets_0, integral)
    material_0 = dolfinx_cuas.pack_coefficients([mu2, lmbda2], entities_0)

    entities_1 = dolfinx_cuas.compute_active_entities(mesh, facets_1, integral)
    material_1 = dolfinx_cuas.pack_coefficients([mu2, lmbda2], entities_1)

    # Pack circumradius on each surface
    h_0 = dolfinx_contact.pack_circumradius_facet(mesh, facets_0)
    h_1 = dolfinx_contact.pack_circumradius_facet(mesh, facets_1)

    # Pack gap, normals and test functions on each surface
    gap_0 = contact.pack_gap(0)
    n_0 = contact.pack_ny(0, gap_0)
    test_fn_0 = contact.pack_test_functions(0, gap_0)
    gap_1 = contact.pack_gap(1)
    n_1 = contact.pack_ny(1, gap_1)
    test_fn_1 = contact.pack_test_functions(1, gap_1)

    # Concatenate all coeffs
    coeff_0 = np.hstack([material_0, h_0, gap_0, n_0, test_fn_0])
    coeff_1 = np.hstack([material_1, h_1, gap_1, n_1, test_fn_1])

    # Assemble jacobian
    J_cuas = _fem.form(J)
    kernel_jac = contact.generate_kernel(kt.Jac)

    def create_A():
        return contact.create_matrix(J_cuas)

    def A(x, A):
        u.vector[:] = x.array
        u_opp_0 = contact.pack_u_contact(0, u._cpp_object, gap_0)
        u_opp_1 = contact.pack_u_contact(1, u._cpp_object, gap_1)
        u_0 = dolfinx_cuas.pack_coefficients([u], entities_0)
        u_1 = dolfinx_cuas.pack_coefficients([u], entities_1)
        c_0 = np.hstack([coeff_0, u_0, u_opp_0])
        c_1 = np.hstack([coeff_1, u_1, u_opp_1])
        contact.assemble_matrix(A, [], 0, kernel_jac, c_0, consts)
        contact.assemble_matrix(A, [], 1, kernel_jac, c_1, consts)
        _fem.assemble_matrix(A, J_cuas)

    # assemble rhs
    F_cuas = _fem.form(F)
    kernel_rhs = contact.generate_kernel(kt.Rhs)

    def create_b():
        return _fem.create_vector(F_cuas)

    def F(x, b):
        u.vector[:] = x.array
        u_opp_0 = contact.pack_u_contact(0, u._cpp_object, gap_0)
        u_opp_1 = contact.pack_u_contact(1, u._cpp_object, gap_1)
        u_0 = dolfinx_cuas.pack_coefficients([u], entities_0)
        u_1 = dolfinx_cuas.pack_coefficients([u], entities_1)
        c_0 = np.hstack([coeff_0, u_0, u_opp_0])
        c_1 = np.hstack([coeff_1, u_1, u_opp_1])
        contact.assemble_vector(b, 0, kernel_rhs, c_0, consts)
        contact.assemble_vector(b, 1, kernel_rhs, c_1, consts)
        _fem.assemble_vector(b, F_cuas)

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

    # Set initial guess
    if initGuess is None:
        u.x.array[:] = 0
    else:
        u.x.array[:] = initGuess.x.array[:]

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
