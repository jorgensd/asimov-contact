from IPython import embed
import dolfinx_contact
from dolfinx_contact.cpp import add_ghost_cells, update_ghosts
import dolfinx.fem
import dolfinx.mesh
import dolfinx_cuas
from mpi4py import MPI
import numpy as np
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 5)


def locator(x):
    return x[0] <= 0.5 + 1e-15


cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, locator)
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
cells = cells[cells < num_cells_local]
cmap = mesh.topology.index_map(mesh.topology.dim)
glob_cells = cmap.local_to_global(cells)
mt = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim, cells, np.full(cells.size, 1,
                                                                   dtype=np.int32))
print(mesh.comm.rank, cmap.ghosts, cmap.ghost_owner_rank())
mesh.topology.create_connectivity(mesh.topology.dim, 0)
partitioner = add_ghost_cells(mesh, cells)
print(partitioner)
embed()
print(update_ghosts(mesh, partitioner))
