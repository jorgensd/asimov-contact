# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Dict, Tuple

import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.la as _la
import dolfinx.mesh as dmesh
import numpy as np
import ufl
from petsc4py import PETSc as _PETSc

from dolfinx_contact.helpers import (NonlinearPDE_SNESProblem, epsilon,
                                     lame_parameters, rigid_motions_nullspace,
                                     sigma_func)


def snes_solver(mesh: dmesh.Mesh, mesh_data: Tuple[dmesh.MeshTagsMetaClass, int, int],
                physical_parameters: dict = {}, plane_loc: float = 0.0, vertical_displacement: float = -0.1,
                quadrature_degree: int = 5, form_compiler_params: Dict = {},
                jit_params: Dict = {}, petsc_options: Dict = {}, snes_options: Dict = {}) -> _fem.Function:
    """
    Solving contact problem against a rigid plane with gap -g from y=0 using PETSc SNES solver

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
    plane_loc
        The location of the plane in y-coordinate (2D) and z-coordinate (3D)
    vertical_displacement
        The amount of verticial displacment enforced on Dirichlet boundary
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
    snes_options
        Parameters to pass to the snes solver, see the `SNES-documentation
        <https://petsc.org/release/docs/manualpages/SNES/SNESSetFromOptions.html>`
    """
    # Compute lame parameters
    plane_strain = physical_parameters.get("strain", False)
    E = physical_parameters.get("E", 1e3)
    nu = physical_parameters.get("nu", 0.1)
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    (facet_marker, top_value, _) = mesh_data
    assert(facet_marker.dim == mesh.topology.dim - 1)

    # function space and problem parameters
    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))  # function space

    # Functions for penalty term. Not used at the moment.
    # def gap(u): # Definition of gap function
    #     x = ufl.SpatialCoordinate(mesh)
    #     return x[1]+u[1]-g
    # def maculay(x): # Definition of Maculay bracket
    #     return (x+abs(x))/2

    # elasticity variational formulation no contact
    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    zero = np.asarray([0, ] * mesh.geometry.dim, dtype=np.float64)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - \
        ufl.inner(_fem.Constant(mesh, zero), v) * dx

    # Stored strain energy density (linear elasticity model)    # penalty = 0
    # psi = 1/2*ufl.inner(sigma(u), epsilon(u))
    # Pi = psi*dx #+ 1/2*(penalty*E/h)*ufl.inner(maculay(-gap(u)),maculay(-gap(u)))*ds(1)

    # # Compute first variation of Pi (directional derivative about u in the direction of v)
    # # Yields same F as above if penalty = 0 and body force 0
    # F = ufl.derivative(Pi, u, v)

    # Dirichlet boundary conditions
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
    bc = _fem.dirichletbc(u_D, dirichlet_dofs)
    # bcs = [bc]

    # create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, u, bc, jit_params=jit_params,
                                       form_compiler_params=form_compiler_params)

    # Inequality constraints (contact constraints)
    # The displacement u must be such that the current configuration x+u
    # remains in the box [xmin = -inf,xmax = inf] x [ymin = -g,ymax = inf]
    # inf replaced by large number for implementation
    lims = np.zeros(2 * mesh.geometry.dim)
    for i in range(mesh.geometry.dim):
        lims[2 * i] = -1e7
        lims[2 * i + 1] = 1e7
    lims[-2] = -plane_loc

    def _constraint_u(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(mesh.geometry.dim):
            values[i] = lims[2 * i + 1] - x[i]
        return values

    def _constraint_l(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(mesh.geometry.dim):
            values[i] = lims[2 * i] - x[i]
        return values

    umax = _fem.Function(V)
    umax.interpolate(_constraint_u)
    umin = _fem.Function(V)
    umin.interpolate(_constraint_l)

    # Create LHS matrix and RHS vector
    b = _la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J = _fem.petsc.create_matrix(problem.a)

    # Create semismooth Newton solver (SNES)
    snes = _PETSc.SNES().create()

    # Set SNES options
    opts = _PETSc.Options()
    snes.setOptionsPrefix(f"snes_solve_{id(snes)}")
    option_prefix = snes.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in snes_options.items():
        opts[k] = v
    opts.prefixPop()
    snes.setFromOptions()

    # Set solve functions and variable bounds
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)
    snes.setVariableBounds(umin.vector, umax.vector)
    null_space = rigid_motions_nullspace(V)
    J.setNearNullSpace(null_space)

    # Set ksp options
    ksp = snes.ksp
    ksp.setOptionsPrefix(f"snes_ksp_{id(ksp)}")
    opts = _PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - plane_loc
        return values

    u.interpolate(_u_initial)
    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    with _common.Timer(f"{num_dofs_global} Solve SNES"):
        snes.solve(None, u.vector)
    u.x.scatter_forward()

    assert(snes.getConvergedReason() > 1)
    assert(snes.getConvergedReason() < 4)

    return u
