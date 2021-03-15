from typing import List
import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import dolfinx.geometry
import numpy as np
import os
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.mesh import CellType
import argparse


class NonlinearPDEProblem:
    """Nonlinear problem class for solving the non-linear problem
    F(u, v) = 0 for all v in V
    """

    def __init__(self, F: ufl.form.Form, u: dolfinx.Function,
                 bcs: List[dolfinx.DirichletBC]):
        """
        Input:
        - F: The PDE residual F(u, v)
        - u: The unknown
        - bcs: List of Dirichlet boundary conditions
        This class set up structures for solving the non-linear problem using Newton's method,
        dF/du(u) du = -F(u)
        """
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = F
        # Create the Jacobian matrix, dF/du
        self.a = ufl.derivative(F, u, du)
        self.bcs = bcs

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.matrix = dolfinx.fem.create_matrix(self.a)
        self.vector = dolfinx.fem.create_vector(self.L)

    def form(self, x: PETSc.Vec):
        """
        This function is called before the residual or Jacobian is computed. This is usually used to update ghost values.
        Input:
           x: The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.
        Input:
           x: The vector containing the latest solution
           b: Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b, self.L)
        # Apply boundary condition
        dolfinx.fem.apply_lifting(b, [self.a], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble the Jacobian matrix.
        Input:
          - x: The vector containing the latest solution
          - A: The matrix to assemble the Jacobian into
        """
        A.zeroEntries()
        dolfinx.fem.assemble_matrix(A, self.a, self.bcs)
        A.assemble()


def solve_manufactured(nx, ny, theta, gamma, linear_solver, plane_strain, nitsche):
    L = 47
    H = 2.73
    mesh = dolfinx.RectangleMesh(
        MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, H, 0])], [nx, ny],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

    def left(x):
        return np.isclose(x[0], 0)

    def top(x):
        return np.isclose(x[1], H)

    tdim = mesh.topology.dim
    left_marker = int(1)
    top_marker = int(2)
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
    left_values = np.full(len(left_facets), left_marker, dtype=np.int32)
    top_values = np.full(len(top_facets), top_marker, dtype=np.int32)
    indices = np.concatenate([left_facets, top_facets])
    values = np.hstack([left_values, top_values])
    # Sort values to work in parallel
    sorted = np.argsort(indices)
    facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices[sorted], values[sorted])

    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 2))
    n = ufl.FacetNormal(mesh)
    E = 1e5
    nu = 0.3
    h = 2 * ufl.Circumradius(mesh)
    mu = E / (2 * (1 + nu))
    if plane_strain:
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # Plane strain
    else:
        lmbda = E * nu / ((1 + nu) * (1 - nu))  # Plane stress

    rho_g = 1e-3
    f = ufl.as_vector((0, 0))
    g = ufl.as_vector((0, -rho_g))

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return 2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))

    u = ufl.TrialFunction(V) if linear_solver else dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(f, v) * dx - ufl.inner(g, v) * ds(top_marker)
    if nitsche:
       # Nitsche for Dirichlet, theta-scheme.
       # https://www.sciencedirect.com/science/article/pii/S004578251830269X
        u_D = ufl.as_vector((0, 0))
        n_facet = ufl.FacetNormal(mesh)
        F += -ufl.inner(sigma(u) * n_facet, v) * ds(left_marker)\
            - theta * ufl.inner(sigma(v) * n_facet, u - u_D) * ds(left_marker)\
            + gamma / h * ufl.inner(u - u_D, v) * ds(left_marker)
        bcs = []
    else:
        # Strong Dirichlet enforcement via lifting
        u_bc = dolfinx.Function(V)
        with u_bc.vector.localForm() as loc:
            loc.set(0)
        bc = dolfinx.DirichletBC(u_bc, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets))
        bcs = [bc]

    # Solve as linear problem
    if linear_solver:
        problem = dolfinx.fem.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=bcs,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
    else:
        # Create nonlinear problem
        problem = NonlinearPDEProblem(F, u, bcs)

        # Create Newton solver
        solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)

        # Set Newton solver options
        solver.atol = 1e-6
        solver.rtol = 1e-6
        solver.convergence_criterion = "incremental"

        # # Set non-linear problem for Newton solver
        solver.setF(problem.F, problem.vector)
        solver.setJ(problem.J, problem.matrix)
        solver.set_form(problem.form)

        # Solve non-linear problem
        n, converged = solver.solve(u.vector)
        assert (converged)
        print(f"Number of interations: {n:d}")

    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

    os.system("mkdir -p results")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_{nx}_{ny}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    # Evaluate deflection at L, H/2
    cells = []
    points_on_proc = []
    point = np.array([L, H / 2, 0])
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point)
    # Choose one of the cells that contains the point
    cell = dolfinx.geometry.select_colliding_cells(mesh, cell_candidates, point, 1)
    # Only use evaluate for points on current processor
    if len(cell) == 1:
        points_on_proc.append(point)
        cells.append(cell[0])
    if len(points_on_proc) > 0:
        u_at_point = u.eval(points_on_proc, cells)

        # Exact solution of Euler Bernoulli:
        # https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Cantilever_beams
        # Second order moment of area for rectangular beam
        I = H**3 / (12)
        uz = (rho_g * L**4) / (8 * E * I)
        print(f"-----{nx}x{ny}--Nitsche:{nitsche} (gamma: {gamma})----\n", f"Maximal deflection: {u_at_point[mesh.geometry.dim-1]}\n",
              f"Theoretical deflection: {-uz}\n", f"Error: {100*abs((u_at_point[mesh.geometry.dim-1]+uz)/uz)} %", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like")
    parser.add_argument("--gamma", default=1000, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    _solve = parser.add_mutually_exclusive_group(required=False)
    _solve.add_argument('--linear', dest='linear_solver', action='store_true',
                        help="Use linear solver", default=False)
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument('--strain', dest='plane_strain', action='store_true',
                         help="Use plane strain formulation", default=False)
    _dirichlet = parser.add_mutually_exclusive_group(required=False)
    _dirichlet.add_argument('--dirichlet', dest='dirichlet', action='store_true',
                            help="Use strong Dirichlet formulation", default=False)

    args = parser.parse_args()
    theta = args.theta
    gamma = args.gamma
    plane_strain = args.plane_strain
    linear_solver = args.linear_solver
    nitsche = not args.dirichlet
    Nx = np.asarray([5 * 2**i for i in range(1, 8)], dtype=np.int32)
    Ny = np.asarray([2**i for i in range(1, 8)], dtype=np.int32)
    for (nx, ny) in zip(Nx, Ny):
        solve_manufactured(nx, ny, theta, gamma, linear_solver, plane_strain, nitsche)
