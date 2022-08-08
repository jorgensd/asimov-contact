# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple

import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.log as _log
import dolfinx_cuas
import numpy as np
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32
import dolfinx.cpp as _cpp
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (rigid_motions_nullspace_subdomains, sigma_func)

kt = dolfinx_contact.cpp.Kernel

__all__ = ["nitsche_unbiased"]


def nitsche_unbiased(ufl_form: ufl.Form, u: _fem.Function, markers: list[_cpp.mesh.MeshTags_int32],
                     contact_data: Tuple[AdjacencyList_int32, list[Tuple[int, int]]],
                     bcs: list[_fem.DirichletBCMetaClass],
                     problem_parameters: dict[str, np.float64],
                     quadrature_degree: int = 5, form_compiler_params: dict = None, jit_params: dict = None,
                     petsc_options: dict = None, newton_options: dict = None,
                     outfile: str = None) -> Tuple[_fem.Function, int, int, float]:
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    F The residual without contact contributions
    J The Jacobian without contact contributions
    u The function to be solved for. Also serves as initial value.
    markers
        A list of meshtags. The first element must mark all separate objects in order to create the correct nullspace.
        The second element must contain the mesh_tags for all puppet surfaces,
        Dirichlet-surfaces and Neumann-surfaces
        All further elements may contain candidate_surfaces
    contact_data = (surfaces, contact_pairs), where
        surfaces: Adjacency list. Links of i are meshtag values for contact
                  surfaces in ith mesh_tag in mesh_tags
        contact_pairs: list of pairs (i, j) marking the ith surface as a puppet
                  surface and the jth surface as the corresponding candidate
                  surface
    problem_parameters
        Dictionary with lame parameters and Nitsche parameters.
        Valid (key, value) tuples are: ('gamma': float), ('theta', float), ('mu', float),
        (lambda, float),
        where theta can be -1, 0 or 1 for skew-symmetric, penalty like or symmetric
        enforcement of Nitsche conditions
    quadrature_degree
        The quadrature degree to use for the custom contact kernels
    form_compiler_params
        Parameters used in FFCX compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINX.
    jit_params
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
    outfile
        File to append solver summary

    """
    form_compiler_params = {} if form_compiler_params is None else form_compiler_params
    jit_params = {} if jit_params is None else jit_params
    petsc_options = {} if petsc_options is None else petsc_options
    newton_options = {} if newton_options is None else newton_options

    if problem_parameters.get("mu") is None:
        raise RuntimeError("Need to supply lame paramters")
    else:
        mu = mu = problem_parameters.get("mu")

    if problem_parameters.get("lambda") is None:
        raise RuntimeError("Need to supply lame paramters")
    else:
        lmbda = problem_parameters.get("lambda")
    if problem_parameters.get("theta") is None:
        raise RuntimeError("Need to supply theta for Nitsche's method")
    else:
        theta = problem_parameters["theta"]
    if problem_parameters.get("gamma") is None:
        raise RuntimeError("Need to supply gamma for Nitsche's method")
    else:
        gamma = problem_parameters.get("gamma")
    sigma = sigma_func(mu, lmbda)

    # Contact data
    contact_pairs = contact_data[1]
    contact_surfaces = contact_data[0]

    # Mesh, function space and FEM functions
    V = u.function_space
    mesh = V.mesh
    v = ufl_form.arguments()[0]  # Test function
    w = ufl.TrialFunction(V)     # Trial function

    h = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    # Integration measure and ufl part of linear/bilinear form
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=markers[1])

    # ufl part of contact
    F = ufl_form
    for contact_pair in contact_pairs:
        surface_value = int(contact_surfaces.links(0)[contact_pair[0]])
        F += - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
            ds(surface_value)
    J = ufl.derivative(F, u, w)

    # Custom assembly
    # create contact class
    with _common.Timer("~Contact: Init"):
        # mode = dolfinx_contact.cpp.ContactMode.Raytracing
        mode = dolfinx_contact.cpp.ContactMode.ClosestPoint
        contact = dolfinx_contact.cpp.Contact(markers[1:], contact_surfaces, contact_pairs,
                                              V._cpp_object, quadrature_degree=quadrature_degree,
                                              search_method=mode)
    with _common.Timer("~Contact: Distance maps"):
        for i in range(len(contact_pairs)):
            contact.create_distance_map(i)

    # pack constants
    consts = np.array([gamma, theta])

    # Pack material parameters mu and lambda on each contact surface
    with _common.Timer("~Contact: Interpolate coeffs (mu, lmbda)"):
        V2 = _fem.FunctionSpace(mesh, ("DG", 0))
        lmbda2 = _fem.Function(V2)
        lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
        mu2 = _fem.Function(V2)
        mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))

    entities = []
    with _common.Timer("~Contact: Compute active entities"):
        for pair in contact_pairs:
            entities.append(contact.active_entities(pair[0]))

    material = []
    with _common.Timer("~Contact: Pack coeffs (mu, lmbda"):
        for i in range(len(contact_pairs)):
            material.append(np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
                mu2._cpp_object, 0, entities[i]),
                dolfinx_contact.cpp.pack_coefficient_quadrature(
                lmbda2._cpp_object, 0, entities[i])]))

    # Pack celldiameter on each surface
    h_packed = []
    with _common.Timer("~Contact: Compute and pack celldiameter"):
        surface_cells = np.unique(np.hstack([entities[i][:, 0] for i in range(len(contact_pairs))]))
        h_int = _fem.Function(V2)
        expr = _fem.Expression(h, V2.element.interpolation_points())
        h_int.interpolate(expr, surface_cells)
        for i in range(len(contact_pairs)):
            h_packed.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                h_int._cpp_object, 0, entities[i]))

    # Pack gap, normals and test functions on each surface
    gaps = []
    normals = []
    test_fns = []
    with _common.Timer("~Contact: Pack gap, normals, testfunction"):
        for i in range(len(contact_pairs)):
            gaps.append(contact.pack_gap(i))
            if mode == dolfinx_contact.cpp.ContactMode.Raytracing:
                normals.append(-contact.pack_nx(i))
            else:
                normals.append(contact.pack_ny(i))
            test_fns.append(contact.pack_test_functions(i))

    # Concatenate all coeffs
    coeffs_const = []
    for i in range(len(contact_pairs)):
        coeffs_const.append(np.hstack([material[i], h_packed[i], gaps[i], normals[i], test_fns[i]]))

    # Generate Jacobian data structures
    J_custom = _fem.form(J, form_compiler_params=form_compiler_params, jit_params=jit_params)
    with _common.Timer("~Contact: Generate Jacobian kernel"):
        kernel_jac = contact.generate_kernel(kt.Jac)
    with _common.Timer("~Contact: Create matrix"):
        J = contact.create_matrix(J_custom)

    # Generate residual data structures
    F_custom = _fem.form(F, form_compiler_params=form_compiler_params, jit_params=jit_params)
    with _common.Timer("~Contact: Generate residual kernel"):
        kernel_rhs = contact.generate_kernel(kt.Rhs)
    with _common.Timer("~Contact: Create vector"):
        b = _fem.petsc.create_vector(F_custom)

    @_common.timed("~Contact: Update coefficients")
    def compute_coefficients(x, coeffs):
        u.vector[:] = x.array
        u_candidate = []
        with _common.Timer("~~Contact: Pack u contact"):
            for i in range(len(contact_pairs)):
                u_candidate.append(contact.pack_u_contact(i, u._cpp_object, gaps[i]))
        u_puppet = []
        with _common.Timer("~~Contact: Pack u"):
            for i in range(len(contact_pairs)):
                u_puppet.append(dolfinx_cuas.pack_coefficients([u], entities[i]))
        for i in range(len(contact_pairs)):
            c_0 = np.hstack([coeffs_const[i], u_puppet[i], u_candidate[i]])
            coeffs[i][:, :] = c_0[:, :]

    @_common.timed("~Contact: Assemble residual")
    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        with _common.Timer("~~Contact: Contact contributions (in assemble vector)"):
            for i in range(len(contact_pairs)):
                contact.assemble_vector(b, i, kernel_rhs, coeffs[i], consts)
        with _common.Timer("~~Contact: Standard contributions (in assemble vector)"):
            _fem.petsc.assemble_vector(b, F_custom)

        # Apply boundary condition
        if len(bcs) > 0:
            _fem.petsc.apply_lifting(b, [J_custom], bcs=[bcs], x0=[x], scale=-1.0)
            b.ghostUpdate(addv=_PETSc.InsertMode.ADD, mode=_PETSc.ScatterMode.REVERSE)
            _fem.petsc.set_bc(b, bcs, x, -1.0)

    @_common.timed("~Contact: Assemble matrix")
    def compute_jacobian_matrix(x, A, coeffs):
        A.zeroEntries()
        with _common.Timer("~~Contact: Contact contributions (in assemble matrix)"):
            for i in range(len(contact_pairs)):
                contact.assemble_matrix(A, [], i, kernel_jac, coeffs[i], consts)
        with _common.Timer("~~Contact: Standard contributions (in assemble matrix)"):
            _fem.petsc.assemble_matrix(A, J_custom, bcs=bcs)
        A.assemble()

    # coefficient arrays
    num_coeffs = contact.coefficients_size(False)
    coeffs = np.array([np.zeros((len(entities[i]), num_coeffs)) for i in range(len(contact_pairs))])
    newton_solver = dolfinx_contact.NewtonSolver(mesh.comm, J, b, coeffs)

    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(V, markers[0], np.unique(markers[0].values))
    newton_solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    newton_solver.set_newton_options(newton_options)

    # Set Krylov solver options
    newton_solver.set_krylov_options(petsc_options)

    dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    _log.set_log_level(_log.LogLevel.OFF)
    # Solve non-linear problem
    timing_str = f"~Contact: {id(dofs_global)} Solve Nitsche"
    with _common.Timer(timing_str):
        n, converged = newton_solver.solve(u)

    if outfile is not None:
        viewer = _PETSc.Viewer().createASCII(outfile, "a")
        newton_solver.krylov_solver.view(viewer)
    newton_time = _common.timing(timing_str)
    if not converged:
        print("Newton solver did not converge")
    u.x.scatter_forward()

    print(f"{dofs_global}\n Number of Newton iterations: {n:d}\n",
          f"Number of Krylov iterations {newton_solver.krylov_iterations}\n", flush=True)
    return u, n, newton_solver.krylov_iterations, newton_time[1]
